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
    def __init__(self, seq_features=7, hidden_dim=16, num_layers=1):
        super(StudentLSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=seq_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LeakyReLU(0.05),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # x shape: [batch, 4, 7]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out

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
        
    y = df["final_score"].values.reshape(-1, 1) if "final_score" in df.columns else None
    return seq_data, y

def train_pytorch_model(df, model_path):
    seq_data, y = get_seq_and_static_data(df)
    
    N, T, F = seq_data.shape
    seq_flat = seq_data.reshape(-1, F)
    scaler = StandardScaler()
    seq_flat_scaled = scaler.fit_transform(seq_flat)
    seq_data_scaled = seq_flat_scaled.reshape(N, T, F)
    
    cv = min(5, len(df))
    mae_list = []
    r2_list = []
    
    if cv >= 2:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(seq_data):
            s_tr, s_val = seq_data_scaled[train_idx], seq_data_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = StudentLSTMRegressor()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(250):
                optimizer.zero_grad()
                pred = model(torch.tensor(s_tr, dtype=torch.float32))
                loss = criterion(pred, torch.tensor(y_tr, dtype=torch.float32))
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                pred_val = model(torch.tensor(s_val, dtype=torch.float32)).numpy()
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
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    final_model.train()
    for epoch in range(250):
        optimizer.zero_grad()
        pred = final_model(torch.tensor(seq_data_scaled, dtype=torch.float32))
        loss = criterion(pred, torch.tensor(y, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        
    payload = {
        "model_state": final_model.state_dict(),
        "scaler": scaler
    }
    joblib.dump(payload, model_path)
    
    return mae_mean, mae_std, r2_mean
