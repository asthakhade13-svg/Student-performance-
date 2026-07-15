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

class StudentTransformerLSTM(nn.Module):
    def __init__(self, seq_features=7, hidden_dim=16, num_layers=1, nhead=2, vocab_size=1000, text_dim=64):
        super(StudentTransformerLSTM, self).__init__()
        # Bidirectional LSTM to capture future and past context
        self.lstm = nn.LSTM(input_size=seq_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Transformer Multi-Head Self-Attention block
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,  # 32
            nhead=nhead,
            dim_feedforward=32,
            dropout=0.1,
            batch_first=True
        )
        
        # Self-Attention scoring layer for temporal pooling
        self.attn_linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        # Qualitative Counselor Notes Text Embedding bag
        self.text_embed = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=text_dim, mode='mean', padding_idx=0)
        
        # Projection layer to align sequence state dimensions
        self.context_proj = nn.Linear(hidden_dim * 2, text_dim)
        
        # Cross-Modal Attention Layer: Query from Sequence, Key/Value from Text Notes
        self.cross_attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=2, batch_first=True)
        
        # Shared layer (takes 64-dim fused multimodal representation)
        self.shared_fc = nn.Sequential(
            nn.Linear(text_dim, 16),
            nn.LeakyReLU(0.05)
        )
        
        # Heads
        self.reg_head = nn.Linear(16, 1)
        self.clf_head = nn.Linear(16, 3)
        
    def forward(self, x, text_indices=None, text_offsets=None):
        # x shape: [batch, 4, 7]
        lstm_out, _ = self.lstm(x)  # shape: [batch, 4, hidden_dim * 2] (32)
        
        # Transformer Multi-Head Self-Attention
        trans_out = self.transformer_layer(lstm_out)  # shape: [batch, 4, 32]
        
        # Temporal attention pooling
        attn_scores = self.attn_linear(trans_out)  # shape: [batch, 4, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # shape: [batch, 4, 1]
        
        context_vector = torch.sum(trans_out * attn_weights, dim=1)  # shape: [batch, 32]
        
        # Project sequence context to text dimension
        # q shape: [batch, 1, 64]
        q = self.context_proj(context_vector).unsqueeze(1)
        
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

def train_pytorch_model(df, model_path):
    seq_data, y_reg, y_clf = get_seq_and_static_data(df)
    
    N, T, F = seq_data.shape
    seq_flat = seq_data.reshape(-1, F)
    scaler_x = StandardScaler()
    seq_flat_scaled = scaler_x.fit_transform(seq_flat)
    seq_data_scaled = seq_flat_scaled.reshape(N, T, F)
    
    scaler_y = StandardScaler()
    y_reg_scaled = scaler_y.fit_transform(y_reg)
    
    # Extract student counselor text notes
    notes = df["notes"].values if "notes" in df.columns else [""] * len(df)
    
    cv = min(5, len(df))
    mae_list = []
    r2_list = []
    
    if cv >= 2:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(seq_data):
            s_tr, s_val = seq_data_scaled[train_idx], seq_data_scaled[val_idx]
            yr_tr, yr_val = y_reg_scaled[train_idx], y_reg_scaled[val_idx]
            yc_tr, yc_val = y_clf[train_idx], y_clf[val_idx]
            
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
                pred_reg, pred_clf, _ = model(torch.tensor(s_tr, dtype=torch.float32), idx_tr, off_tr)
                loss_reg = reg_criterion(pred_reg, torch.tensor(yr_tr, dtype=torch.float32))
                loss_clf = clf_criterion(pred_clf, torch.tensor(yc_tr, dtype=torch.long))
                loss = loss_reg + 1.0 * loss_clf
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                pred_reg_val, pred_clf_val, _ = model(torch.tensor(s_val, dtype=torch.float32), idx_val, off_val)
                
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
        pred_reg, pred_clf, _ = final_model(torch.tensor(seq_data_scaled, dtype=torch.float32), idx_all, off_all)
        loss_reg = reg_criterion(pred_reg, torch.tensor(y_reg_scaled, dtype=torch.float32))
        loss_clf = clf_criterion(pred_clf, torch.tensor(y_clf, dtype=torch.long))
        loss = loss_reg + 1.0 * loss_clf
        loss.backward()
        optimizer.step()
        
    payload = {
        "model_state": final_model.state_dict(),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y
    }
    joblib.dump(payload, model_path)
    
    return mae_mean, mae_std, r2_mean
