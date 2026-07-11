import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class StudentAttentionLSTM(nn.Module):
    def __init__(self, seq_features=7, hidden_dim=16, num_layers=1):
        super(StudentAttentionLSTM, self).__init__()
        # Bidirectional LSTM to capture future and past context
        self.lstm = nn.LSTM(input_size=seq_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Self-Attention scoring layer: maps Bi-LSTM hidden state (dim hidden_dim*2 = 32) to a scalar score
        self.attn_linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        # Shared layer
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.LeakyReLU(0.05)
        )
        
        # Heads
        self.reg_head = nn.Linear(16, 1)
        self.clf_head = nn.Linear(16, 3)
        
    def forward(self, x):
        # x shape: [batch, 4, 7]
        lstm_out, _ = self.lstm(x)  # shape: [batch, 4, hidden_dim * 2]
        
        # Compute raw attention scores for each time step
        attn_scores = self.attn_linear(lstm_out)  # shape: [batch, 4, 1]
        
        # Softmax over time steps
        attn_weights = torch.softmax(attn_scores, dim=1)  # shape: [batch, 4, 1]
        
        # Weighted context vector
        context_vector = torch.sum(lstm_out * attn_weights, dim=1)  # shape: [batch, hidden_dim * 2]
        
        shared_out = self.shared_fc(context_vector)
        
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
    
    cv = min(5, len(df))
    mae_list = []
    r2_list = []
    
    if cv >= 2:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(seq_data):
            s_tr, s_val = seq_data_scaled[train_idx], seq_data_scaled[val_idx]
            yr_tr, yr_val = y_reg_scaled[train_idx], y_reg_scaled[val_idx]
            yc_tr, yc_val = y_clf[train_idx], y_clf[val_idx]
            
            model = StudentAttentionLSTM()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            reg_criterion = nn.MSELoss()
            clf_criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(250):
                optimizer.zero_grad()
                pred_reg, pred_clf, _ = model(torch.tensor(s_tr, dtype=torch.float32))
                loss_reg = reg_criterion(pred_reg, torch.tensor(yr_tr, dtype=torch.float32))
                loss_clf = clf_criterion(pred_clf, torch.tensor(yc_tr, dtype=torch.long))
                loss = loss_reg + 1.0 * loss_clf
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                pred_reg_val, pred_clf_val, _ = model(torch.tensor(s_val, dtype=torch.float32))
                
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
        
    final_model = StudentAttentionLSTM()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01, weight_decay=1e-4)
    reg_criterion = nn.MSELoss()
    clf_criterion = nn.CrossEntropyLoss()
    
    final_model.train()
    for epoch in range(250):
        optimizer.zero_grad()
        pred_reg, pred_clf, _ = final_model(torch.tensor(seq_data_scaled, dtype=torch.float32))
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
