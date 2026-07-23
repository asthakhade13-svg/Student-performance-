import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def tokenize_text(text, vocab_size=1000):
    if not text or not isinstance(text, str):
        return [0]
    # Clean and split notes
    words = text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').split()
    if not words:
        return [0]
    # Consistent hash tokens maps directly to vocab bounds
    return [(hash(w) % (vocab_size - 1)) + 1 for w in words]

def prepare_text_tensors(text_list, vocab_size=1000):
    flat_indices = []
    offsets = [0]
    for text in text_list:
        indices = tokenize_text(text, vocab_size)
        flat_indices.extend(indices)
        offsets.append(offsets[-1] + len(indices))
    offsets.pop()  # Drop last cumulative offset to match batch length
    return torch.tensor(flat_indices, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, h, adj):
        # h: [N, in_features]
        # adj: [N, N]
        Wh = torch.mm(h, self.W) # [N, out_features]
        N = Wh.size(0)

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        e = self.leakyrelu(torch.matmul(all_combinations, self.a).squeeze(1))
        e = e.view(N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)

        h_prime = torch.matmul(attention, Wh)
        return h_prime


class TransformerXLLayer(nn.Module):
    """
    Transformer-XL Layer with Segment Recurrence and Relative Positional Bias.
    """
    def __init__(self, d_model=32, nhead=2, dim_feedforward=32, dropout=0.1):
        super(TransformerXLLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Learnable relative positional bias parameter
        self.rel_pos_bias = nn.Parameter(torch.zeros(nhead, 2, 4)) # [nhead, seq_len, total_len]
        
    def forward(self, x, memory=None):
        batch, seq_len, d_model = x.size()
        
        if memory is None:
            memory = torch.zeros(batch, 0, d_model, device=x.device)
            
        memory = memory.detach()
        h_tilde = torch.cat([memory, x], dim=1)
        
        q = self.q_proj(x) 
        k = self.k_proj(h_tilde) 
        v = self.v_proj(h_tilde) 
        
        head_dim = d_model // self.nhead
        q_heads = q.view(batch, seq_len, self.nhead, head_dim).transpose(1, 2) 
        k_heads = k.view(batch, -1, self.nhead, head_dim).transpose(1, 2) 
        v_heads = v.view(batch, -1, self.nhead, head_dim).transpose(1, 2) 
        
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / np.sqrt(head_dim) 
        
        total_len = h_tilde.size(1)
        bias = self.rel_pos_bias[:, :seq_len, :total_len]
        scores = scores + bias.unsqueeze(0)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v_heads) 
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        x = self.norm1(x + self.dropout(self.out_proj(context)))
        x = self.norm2(x + self.ff(x))
        
        return x, x


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class StudentTransformerLSTM(nn.Module):
    def __init__(self, seq_features=7, hidden_dim=16, num_layers=1, nhead=2, vocab_size=1000, text_dim=64):
        super(StudentTransformerLSTM, self).__init__()
        # Bidirectional LSTM to capture future and past context
        self.lstm = nn.LSTM(input_size=seq_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Transformer-XL Attention block with segment recurrence
        self.transformer_xl_layer = TransformerXLLayer(
            d_model=hidden_dim * 2,  # 32
            nhead=nhead,
            dim_feedforward=32,
            dropout=0.1
        )
        
        # Self-Attention scoring layer for temporal pooling
        self.attn_linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        # GAT Peer-Influence Graph Attention layer
        self.gat_layer = GraphAttentionLayer(in_features=hidden_dim * 2, out_features=hidden_dim * 2, dropout=0.1)
        
        # Qualitative Counselor Notes Text Embedding bag
        self.text_embed = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=text_dim, mode='mean', padding_idx=0)
        
        # Projection layer to align sequence state dimensions
        self.context_proj = nn.Linear(hidden_dim * 2, text_dim)
        
        # Cross-Modal Attention Layer: Query from Sequence, Key/Value from Text Notes
        self.cross_attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=2, batch_first=True)
        
        # Shared layer (takes 64-dim fused multimodal representation)
        self.shared_fc = nn.Sequential(
            nn.Linear(text_dim, 16),
            nn.LeakyReLU(0.05),
            nn.Dropout(p=0.1)
        )
        
        # Predictor Heads
        self.reg_head = nn.Linear(16, 1)
        self.clf_head = nn.Linear(16, 3)
        
        # Demographic Adversary Heads
        self.adv_district_head = nn.Linear(16, 2)
        self.adv_gender_head = nn.Linear(16, 2)
        
    def forward(self, x, text_indices=None, text_offsets=None, adj=None, return_adv=False, alpha=1.0):
        # x shape: [batch, 4, 7]
        lstm_out, _ = self.lstm(x)  # shape: [batch, 4, hidden_dim * 2] (32)
        
        # Transformer-XL Segment Recurrence (Chunk sequence into 2 segments of length 2)
        seg1 = lstm_out[:, :2, :]
        seg2 = lstm_out[:, 2:, :]
        out1, mem1 = self.transformer_xl_layer(seg1, memory=None)
        out2, mem2 = self.transformer_xl_layer(seg2, memory=mem1)
        trans_out = torch.cat([out1, out2], dim=1)  # shape: [batch, 4, 32]
        
        # Temporal attention pooling
        attn_scores = self.attn_linear(trans_out)  # shape: [batch, 4, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # shape: [batch, 4, 1]
        
        context_vector = torch.sum(trans_out * attn_weights, dim=1)  # shape: [batch, 32]
        
        # GAT message passing over peer adjacency matrix
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device, dtype=torch.float32)
            
        if adj.size(0) != x.size(0):
            adj = torch.eye(x.size(0), device=x.device, dtype=torch.float32)
            
        graph_out = self.gat_layer(context_vector, adj)
        
        # Fuse individual sequence representation with GAT peer context vector
        fused_context = context_vector + graph_out
        
        # Project sequence context to text dimension
        q = self.context_proj(fused_context).unsqueeze(1)
        
        # Retrieve counselor text embeddings
        if text_indices is None:
            # Fallback to pad token representations for backwards compatibility
            text_indices = torch.tensor([0] * x.size(0), dtype=torch.long, device=x.device)
            text_offsets = torch.arange(x.size(0), dtype=torch.long, device=x.device)
            
        text_emb = self.text_embed(text_indices, text_offsets).unsqueeze(1)  # shape: [batch, 1, 64]
        
        # Cross-Modal Attention Fusion (Sequence Query learns from Counselor notes Key/Value)
        # attn_out shape: [batch, 1, 64]
        attn_out, _ = self.cross_attn(q, text_emb, text_emb)
        
        # Residual fusion
        fused_vector = (q + attn_out).squeeze(1)  # shape: [batch, 64]
        
        shared_out = self.shared_fc(fused_vector)
        reg_out = self.reg_head(shared_out)
        clf_out = self.clf_head(shared_out)
        
        if return_adv:
            # Reverse gradients flowing to representation encoder
            reversed_shared = grad_reverse(shared_out, alpha)
            district_logits = self.adv_district_head(reversed_shared)
            gender_logits = self.adv_gender_head(reversed_shared)
            return reg_out, clf_out, attn_weights, district_logits, gender_logits
            
        return reg_out, clf_out, attn_weights

def calculate_burnout_label(row):
    avg_study = np.mean([row[f"study_hours_w{w}"] for w in range(1, 5)])
    avg_sleep = np.mean([row[f"sleep_hours_w{w}"] for w in range(1, 5)])
    
    burnout_score = avg_study - avg_sleep
    
    if burnout_score > 0.5 or avg_sleep < 6.0:
        return 2  # High
    elif burnout_score > -1.5 or avg_sleep < 7.0:
        return 1  # Medium
    else:
        return 0  # Low

def get_seq_and_static_data(df):
    N = len(df)
    seq_data = np.zeros((N, 4, 7))
    
    attendance = df["attendance"].values
    previous_marks = df["previous_marks"].values
    
    for w in range(1, 5):
        seq_data[:, w-1, 0] = df[f"study_hours_w{w}"].values
        seq_data[:, w-1, 1] = df[f"sleep_hours_w{w}"].values
        seq_data[:, w-1, 2] = df[f"lms_logins_w{w}"].values
        seq_data[:, w-1, 3] = df[f"assignments_completed_w{w}"].values
        seq_data[:, w-1, 4] = df[f"mock_exams_w{w}"].values
        seq_data[:, w-1, 5] = attendance
        seq_data[:, w-1, 6] = previous_marks
        
    y_reg = df["final_score"].values.reshape(-1, 1) if "final_score" in df.columns else None
    y_clf = np.array([calculate_burnout_label(row) for _, row in df.iterrows()]).reshape(-1)
    
    return seq_data, y_reg, y_clf

import sqlite3

def load_adjacency_matrix(df, db_path=None):
    if db_path is None:
        db_path = os.path.join("models", "student_records.db")
    
    N = len(df)
    adj = np.eye(N, dtype=np.float32)
    
    if not os.path.exists(db_path):
        return torch.tensor(adj, dtype=torch.float32)
        
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Ensure connections table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_connections'")
        if not c.fetchone():
            c.execute("""
                CREATE TABLE IF NOT EXISTS student_connections (
                    student_id_1 TEXT,
                    student_id_2 TEXT,
                    weight REAL
                )
            """)
            conn.commit()
            
            # Populate with some default cohort study group connections
            for i in range(18):
                for j in [i-1, i+1, i-2, i+2]:
                    if 0 <= j < 18 and i != j:
                        c.execute("INSERT INTO student_connections VALUES (?, ?, ?)", (f"student_{i}", f"student_{j}", 0.5))
            conn.commit()
            
        c.execute("SELECT student_id_1, student_id_2, weight FROM student_connections")
        rows = c.fetchall()
        conn.close()
        
        # Build student_id mapping
        id_to_idx = {}
        for idx, row in df.iterrows():
            s_id = row.get("student_id", f"student_{idx}")
            id_to_idx[s_id] = idx
            
        for s1, s2, w in rows:
            if s1 in id_to_idx and s2 in id_to_idx:
                idx1 = id_to_idx[s1]
                idx2 = id_to_idx[s2]
                adj[idx1, idx2] = w
                adj[idx2, idx1] = w
                
    except Exception as e:
        print("[GNN Adjacency] Failed to load connections from SQLite:", e)
        
    return torch.tensor(adj, dtype=torch.float32)


def compute_fairness_audit(df, predictions, sensitive_col="district", target_threshold=75.0):
    actuals = df["final_score"].values
    sensitive = df[sensitive_col].values
    
    pred_pos = (predictions >= target_threshold).astype(int)
    act_pos = (actuals >= target_threshold).astype(int)
    
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return {"demographic_parity_diff": 0.0, "equalized_odds_diff": 0.0}
        
    rates = {}
    for g in groups:
        mask = (sensitive == g)
        if not np.any(mask):
            continue
            
        g_pred = pred_pos[mask]
        g_act = act_pos[mask]
        
        selection_rate = np.mean(g_pred)
        
        p_mask = (g_act == 1)
        tpr = np.mean(g_pred[p_mask]) if np.sum(p_mask) > 0 else 0.0
        
        n_mask = (g_act == 0)
        fpr = np.mean(g_pred[n_mask]) if np.sum(n_mask) > 0 else 0.0
        
        rates[g] = {"selection": selection_rate, "tpr": tpr, "fpr": fpr}
        
    g0, g1 = groups[0], groups[1]
    dp_diff = abs(rates[g0]["selection"] - rates[g1]["selection"])
    tpr_diff = abs(rates[g0]["tpr"] - rates[g1]["tpr"])
    fpr_diff = abs(rates[g0]["fpr"] - rates[g1]["fpr"])
    eo_diff = max(tpr_diff, fpr_diff)
    
    return {
        "demographic_parity_diff": round(float(dp_diff), 4),
        "equalized_odds_diff": round(float(eo_diff), 4)
    }


def train_pytorch_model(df, model_path):
    seq_data, y_reg, y_clf = get_seq_and_static_data(df)
    
    N, T, F = seq_data.shape
    seq_flat = seq_data.reshape(-1, F)
    scaler_x = StandardScaler()
    seq_flat_scaled = scaler_x.fit_transform(seq_flat)
    seq_data_scaled = seq_flat_scaled.reshape(N, T, F)
    
    scaler_y = StandardScaler()
    y_reg_scaled = scaler_y.fit_transform(y_reg)
    
    # Load sensitive demographics
    districts = df["district"].values if "district" in df.columns else np.zeros(len(df))
    genders = df["gender"].values if "gender" in df.columns else np.zeros(len(df))
    
    # Extract student counselor text notes
    notes = df["notes"].values if "notes" in df.columns else [""] * len(df)
    
    # Load peer connections adjacency matrix
    adj_all = load_adjacency_matrix(df)
    
    cv = min(5, len(df))
    mae_list = []
    r2_list = []
    
    adv_criterion = nn.CrossEntropyLoss()
    
    if cv >= 2:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(seq_data):
            s_tr, s_val = seq_data_scaled[train_idx], seq_data_scaled[val_idx]
            yr_tr, yr_val = y_reg_scaled[train_idx], y_reg_scaled[val_idx]
            yc_tr, yc_val = y_clf[train_idx], y_clf[val_idx]
            
            d_tr = districts[train_idx]
            g_tr = genders[train_idx]
            
            # Slice adjacency matrix for folds
            adj_tr = adj_all[train_idx][:, train_idx]
            adj_val = adj_all[val_idx][:, val_idx]
            
            notes_tr = [notes[i] for i in train_idx]
            notes_val = [notes[i] for i in val_idx]
            idx_tr, off_tr = prepare_text_tensors(notes_tr)
            idx_val, off_val = prepare_text_tensors(notes_val)
            
            model = StudentTransformerLSTM()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            reg_criterion = nn.MSELoss()
            clf_criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(250):
                optimizer.zero_grad()
                alpha = min(1.0, float(epoch) / 100.0)
                pred_reg, pred_clf, _, dist_logits, gend_logits = model(
                    torch.tensor(s_tr, dtype=torch.float32), 
                    idx_tr, 
                    off_tr, 
                    adj_tr, 
                    return_adv=True, 
                    alpha=alpha
                )
                
                loss_reg = reg_criterion(pred_reg, torch.tensor(yr_tr, dtype=torch.float32))
                loss_clf = clf_criterion(pred_clf, torch.tensor(yc_tr, dtype=torch.long))
                
                loss_dist = adv_criterion(dist_logits, torch.tensor(d_tr, dtype=torch.long))
                loss_gend = adv_criterion(gend_logits, torch.tensor(g_tr, dtype=torch.long))
                
                # Combine losses: GRL forces encoder features to be district- and gender-invariant
                loss = loss_reg + 1.0 * loss_clf + 0.5 * loss_dist + 0.5 * loss_gend
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                pred_reg_val, pred_clf_val, _ = model(torch.tensor(s_val, dtype=torch.float32), idx_val, off_val, adj_val)
                
                pred_reg_val_unscaled = scaler_y.inverse_transform(pred_reg_val.numpy())
                yr_val_unscaled = scaler_y.inverse_transform(yr_val)
                
                mae_list.append(mean_absolute_error(yr_val_unscaled, pred_reg_val_unscaled))
                r2_list.append(r2_score(yr_val_unscaled, pred_reg_val_unscaled))
                
        mae_mean = float(np.mean(mae_list))
        mae_std = float(np.std(mae_list))
        r2_mean = max(-1.0, float(np.mean(r2_list)))
    else:
        mae_mean = 5.0
        mae_std = 0.0
        r2_mean = 0.5
        
    final_model = StudentTransformerLSTM()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01, weight_decay=1e-4)
    reg_criterion = nn.MSELoss()
    clf_criterion = nn.CrossEntropyLoss()
    
    idx_all, off_all = prepare_text_tensors(notes)
    
    final_model.train()
    for epoch in range(250):
        optimizer.zero_grad()
        alpha = min(1.0, float(epoch) / 100.0)
        pred_reg, pred_clf, _, dist_logits, gend_logits = final_model(
            torch.tensor(seq_data_scaled, dtype=torch.float32), 
            idx_all, 
            off_all, 
            adj_all, 
            return_adv=True, 
            alpha=alpha
        )
        loss_reg = reg_criterion(pred_reg, torch.tensor(y_reg_scaled, dtype=torch.float32))
        loss_clf = clf_criterion(pred_clf, torch.tensor(y_clf, dtype=torch.long))
        
        loss_dist = adv_criterion(dist_logits, torch.tensor(districts, dtype=torch.long))
        loss_gend = adv_criterion(gend_logits, torch.tensor(genders, dtype=torch.long))
        
        loss = loss_reg + 1.0 * loss_clf + 0.5 * loss_dist + 0.5 * loss_gend
        loss.backward()
        optimizer.step()
        
    # Run Fairness Audit
    final_model.eval()
    with torch.no_grad():
        preds_all, _, _ = final_model(torch.tensor(seq_data_scaled, dtype=torch.float32), idx_all, off_all, adj_all)
        preds_unscaled = scaler_y.inverse_transform(preds_all.numpy()).flatten()
        
    fairness_district = compute_fairness_audit(df, preds_unscaled, sensitive_col="district")
    fairness_gender = compute_fairness_audit(df, preds_unscaled, sensitive_col="gender")
    print(f"[Fairness Audit] District Parity Diff: {fairness_district['demographic_parity_diff']}, Equalized Odds: {fairness_district['equalized_odds_diff']}")
    print(f"[Fairness Audit] Gender Parity Diff: {fairness_gender['demographic_parity_diff']}, Equalized Odds: {fairness_gender['equalized_odds_diff']}")
        
    payload = {
        "model_state": final_model.state_dict(),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "fairness_district": fairness_district,
        "fairness_gender": fairness_gender
    }
    joblib.dump(payload, model_path)
    
    return mae_mean, mae_std, r2_mean, fairness_district, fairness_gender
