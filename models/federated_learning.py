import os
import sqlite3
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.lstm_model import StudentTransformerLSTM, get_seq_and_static_data
import joblib
import json
from datetime import datetime

DB_PATH = os.path.join("models", "student_records.db")
REGISTRY_PATH = os.path.join("models", "registry.json")
MODELS_DIR = "models"
FEATURE_COLS = ["attendance", "previous_marks"]
for w in range(1, 5):
    FEATURE_COLS.extend([
        f"study_hours_w{w}",
        f"sleep_hours_w{w}",
        f"lms_logins_w{w}",
        f"assignments_completed_w{w}",
        f"mock_exams_w{w}"
    ])

def load_school_data(school_id):
    conn = sqlite3.connect(DB_PATH)
    table_map = {
        "alpha": "student_data_alpha",
        "beta": "student_data_beta",
        "gamma": "student_data_gamma"
    }
    table_name = table_map.get(school_id.lower(), "student_data_alpha")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def train_local_model(global_state, df, scaler_x, scaler_y, epochs=10, lr=0.01, noise_scale=0.01):
    seq_data, y_reg, y_clf = get_seq_and_static_data(df)
    N, T, F = seq_data.shape
    
    # Scale sequence features
    seq_flat = seq_data.reshape(-1, F)
    seq_flat_scaled = scaler_x.transform(seq_flat)
    seq_data_scaled = seq_flat_scaled.reshape(N, T, F)
    
    # Scale target exam scores
    y_reg_scaled = scaler_y.transform(y_reg)
    
    local_model = StudentTransformerLSTM()
    local_model.load_state_dict(global_state)
    
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
    reg_criterion = nn.MSELoss()
    clf_criterion = nn.CrossEntropyLoss()
    
    local_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_reg, pred_clf, _ = local_model(torch.tensor(seq_data_scaled, dtype=torch.float32))
        loss_reg = reg_criterion(pred_reg, torch.tensor(y_reg_scaled, dtype=torch.float32))
        loss_clf = clf_criterion(pred_clf, torch.tensor(y_clf, dtype=torch.long))
        loss = loss_reg + 1.0 * loss_clf
        loss.backward()
        optimizer.step()
        
    # Differential Privacy: Inject minor Gaussian noise to final parameters before averaging
    if noise_scale > 0:
        with torch.no_grad():
            for param in local_model.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
                
    return local_model.state_dict()

def run_federated_rounds(rounds=3, epochs=10, lr=0.01, noise_scale=0.01):
    logs = []
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Seeding Federated Learning pipeline...")
    
    # 1. Load active scaler states from existing dataset
    conn = sqlite3.connect(DB_PATH)
    all_df = pd.read_sql_query("SELECT * FROM student_data_alpha UNION ALL SELECT * FROM student_data_beta UNION ALL SELECT * FROM student_data_gamma", conn)
    conn.close()
    
    seq_data, y_reg, _ = get_seq_and_static_data(all_df)
    N, T, F = seq_data.shape
    seq_flat = seq_data.reshape(-1, F)
    scaler_x = StandardScaler().fit(seq_flat)
    scaler_y = StandardScaler().fit(y_reg)
    
    # Initialize global model parameters
    global_model = StudentTransformerLSTM()
    global_state = global_model.state_dict()
    
    schools = ["alpha", "beta", "gamma"]
    school_dfs = {s: load_school_data(s) for s in schools}
    
    # Check if there is data
    for s in schools:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Silo School {s.upper()} loaded: {len(school_dfs[s])} student records.")
        
    for r in range(1, rounds + 1):
        logs.append(f"--- Round {r} / {rounds} ---")
        local_states = []
        client_sizes = []
        
        for s in schools:
            df = school_dfs[s]
            if len(df) < 2:
                logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] School {s.upper()} has insufficient data. Skipping node local epoch.")
                continue
            
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] School {s.upper()}: Executing local training round & injecting DP Gaussian noise (scale={noise_scale})...")
            state = train_local_model(global_state, df, scaler_x, scaler_y, epochs=epochs, lr=lr, noise_scale=noise_scale)
            local_states.append(state)
            client_sizes.append(len(df))
            
        if not local_states:
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: No school nodes participated in training.")
            return False, logs, None
            
        # Federated Averaging (FedAvg)
        total_samples = sum(client_sizes)
        client_weights = [c / total_samples for c in client_sizes]
        
        new_global_state = {}
        for key in global_state.keys():
            temp_tensor = torch.zeros_like(global_state[key], dtype=torch.float32)
            for idx, state in enumerate(local_states):
                temp_tensor += client_weights[idx] * state[key].to(torch.float32)
            new_global_state[key] = temp_tensor
            
        global_state = new_global_state
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Federated aggregation complete using FedAvg weights: " + 
                    ", ".join([f"{schools[i].upper()}: {client_weights[i]:.2%}" for i in range(len(client_weights))]))
        
    # Evaluate aggregated global model on full dataset
    global_model.load_state_dict(global_state)
    global_model.eval()
    
    seq_all_scaled = scaler_x.transform(seq_flat).reshape(N, T, F)
    with torch.no_grad():
        pred_reg, _, _ = global_model(torch.tensor(seq_all_scaled, dtype=torch.float32))
        pred_unscaled = scaler_y.inverse_transform(pred_reg.numpy())
        
        mae = float(np.mean(np.abs(y_reg - pred_unscaled)))
        r2 = float(r2_score_fn(y_reg, pred_unscaled))
        
    logs.append(f"--- Federated Aggregation Finished ---")
    logs.append(f"Global model evaluated on full cohort: MAE = {mae:.2f}, R2 = {r2:.2f}")
    
    # Save versioned checkpoint
    # Read registry
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, 'r') as f:
                registry = json.load(f)
        except Exception:
            registry = {"active_version": None, "history": []}
    else:
        registry = {"active_version": None, "history": []}
        
    ver_num = len(registry["history"]) + 1
    version = f"v{ver_num}"
    model_filename = f"model_{version}.pth"
    model_filepath = os.path.join(MODELS_DIR, model_filename)
    
    payload = {
        "model_state": global_state,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y
    }
    joblib.dump(payload, model_filepath)
    
    entry = {
        "version": version,
        "path": model_filepath,
        "r2": round(r2, 2),
        "mae": round(mae, 2),
        "mae_std": 0.0,
        "data_size": total_samples,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (Federated)"
    }
    registry["history"].append(entry)
    registry["active_version"] = version
    
    # Keep only the last 5 version checkpoints
    if len(registry["history"]) > 5:
        for run in registry["history"][:-5]:
            old_path = run.get("path")
            if old_path and os.path.exists(old_path) and run["version"] != registry["active_version"]:
                try:
                    os.remove(old_path)
                except Exception:
                    pass
                    
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)
        
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Federated model version '{version}' registered as active.")
    return True, logs, version

def r2_score_fn(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)
