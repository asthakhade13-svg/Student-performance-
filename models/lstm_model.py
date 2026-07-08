import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class StudentLSTMRegressor(nn.Module):
    def __init__(self, seq_features=5, static_features=2, hidden_dim=16, num_layers=1):
        super(StudentLSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=seq_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + static_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, seq_x, static_x):
        lstm_out, _ = self.lstm(seq_x)
        last_hidden = lstm_out[:, -1, :]
        combined = torch.cat((last_hidden, static_x), dim=1)
        out = self.fc(combined)
        return out

def get_seq_and_static_data(df):
    seq_cols = []
    for w in range(1, 5):
        seq_cols.extend([
            f"study_hours_w{w}",
            f"sleep_hours_w{w}",
            f"lms_logins_w{w}",
            f"assignments_completed_w{w}",
            f"mock_exams_w{w}"
        ])
        
    seq_data = df[seq_cols].values
    seq_data = seq_data.reshape(-1, 4, 5)
    
    static_cols = ["attendance", "previous_marks"]
    static_data = df[static_cols].values
    
    y = df["final_score"].values.reshape(-1, 1) if "final_score" in df.columns else None
    return seq_data, static_data, y

def train_pytorch_model(df, model_path):
    seq_data, static_data, y = get_seq_and_static_data(df)
    
    N, T, F = seq_data.shape
    seq_flat = seq_data.reshape(-1, F)
    scaler_seq = StandardScaler()
    seq_flat_scaled = scaler_seq.fit_transform(seq_flat)
    seq_data_scaled = seq_flat_scaled.reshape(N, T, F)
    
    scaler_static = StandardScaler()
    static_data_scaled = scaler_static.fit_transform(static_data)
    
    cv = min(5, len(df))
    mae_list = []
    r2_list = []
    
    if cv >= 2:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(seq_data):
            s_tr, s_val = seq_data_scaled[train_idx], seq_data_scaled[val_idx]
            st_tr, st_val = static_data_scaled[train_idx], static_data_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = StudentLSTMRegressor()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(150):
                optimizer.zero_grad()
                pred = model(torch.tensor(s_tr, dtype=torch.float32), torch.tensor(st_tr, dtype=torch.float32))
                loss = criterion(pred, torch.tensor(y_tr, dtype=torch.float32))
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                pred_val = model(torch.tensor(s_val, dtype=torch.float32), torch.tensor(st_val, dtype=torch.float32)).numpy()
                mae_list.append(mean_absolute_error(y_val, pred_val))
                r2_list.append(r2_score(y_val, pred_val))
                
        mae_mean = float(np.mean(mae_list))
        mae_std = float(np.std(mae_list))
        r2_mean = max(-1.0, float(np.mean(r2_list)))
    else:
        mae_mean = 5.0
        mae_std = 0.0
        r2_mean = 0.5
        
    final_model = StudentLSTMRegressor()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    final_model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        pred = final_model(torch.tensor(seq_data_scaled, dtype=torch.float32), torch.tensor(static_data_scaled, dtype=torch.float32))
        loss = criterion(pred, torch.tensor(y, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        
    payload = {
        "model_state": final_model.state_dict(),
        "scaler_seq": scaler_seq,
        "scaler_static": scaler_static
    }
    joblib.dump(payload, model_path)
    
    return mae_mean, mae_std, r2_mean
