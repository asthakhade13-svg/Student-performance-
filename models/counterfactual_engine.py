import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from models.lstm_model import get_seq_and_static_data

FEATURE_COLS = ["attendance", "previous_marks"]
for w in range(1, 5):
    FEATURE_COLS.extend([
        f"study_hours_w{w}",
        f"sleep_hours_w{w}",
        f"lms_logins_w{w}",
        f"assignments_completed_w{w}",
        f"mock_exams_w{w}"
    ])

FEATURE_BOUNDS = {
    "attendance": (0.0, 100.0),
    "previous_marks": (0.0, 100.0),
    "study_hours": (0.0, 12.0),
    "sleep_hours": (3.0, 12.0),
    "lms_logins": (0.0, 300.0),
    "assignments_completed": (0.0, 10.0),
    "mock_exams": (0.0, 100.0)
}

def compute_counterfactual_recourse(model, scaler_x, scaler_y, current_features, target_score=85.0, steps=100, lr=0.05):
    """
    Calculates the minimal actionable modification (Counterfactual Recourse) required
    to achieve a target exam score y_target.
    Uses gradient-based optimization on input feature perturbations under realistic domain constraints.
    """
    model.eval()
    
    # 1. Convert current features dict into DataFrame format
    feat_dict = {}
    for col in FEATURE_COLS:
        feat_dict[col] = [float(current_features.get(col, 0.0))]
        
    df_curr = pd.DataFrame(feat_dict)
    seq_curr, _, _ = get_seq_and_static_data(df_curr)  # shape: [1, 4, 7]
    
    # Initial unscaled sequence data: shape [4, 7]
    seq_raw = seq_curr[0]
    
    # Feature indices mapping within seq_raw:
    # 0: study_hours, 1: sleep_hours, 2: lms_logins, 3: assignments_completed, 4: mock_exams, 5: attendance, 6: previous_marks
    
    # We optimize delta parameters for weekly adjustable metrics (study, sleep, lms, assign, mock)
    # delta shape: [4, 5] for the 5 actionable metrics across 4 weeks
    delta = torch.zeros((4, 5), requires_grad=True)
    optimizer = optim.Adam([delta], lr=lr)
    
    target_tensor = torch.tensor([[target_score]], dtype=torch.float32)
    
    best_delta = delta.detach().clone()
    best_loss = float('inf')
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Build candidate sequence
        candidate_seq = torch.tensor(seq_raw, dtype=torch.float32).clone()
        
        # Apply positive/constrained deltas
        # study_hours (col 0): non-negative boost
        candidate_seq[:, 0] = torch.clamp(candidate_seq[:, 0] + torch.relu(delta[:, 0]), 0.0, 12.0)
        
        # sleep_hours (col 1): health-rest target adjustment (towards 7.5 - 8.0 hrs)
        candidate_seq[:, 1] = torch.clamp(candidate_seq[:, 1] + delta[:, 1], 5.5, 9.0)
        
        # lms_logins (col 2): non-negative boost
        candidate_seq[:, 2] = torch.clamp(candidate_seq[:, 2] + torch.relu(delta[:, 2]), 0.0, 300.0)
        
        # assignments_completed (col 3): non-negative boost
        candidate_seq[:, 3] = torch.clamp(candidate_seq[:, 3] + torch.relu(delta[:, 3]), 0.0, 10.0)
        
        # mock_exams (col 4): non-negative boost
        candidate_seq[:, 4] = torch.clamp(candidate_seq[:, 4] + torch.relu(delta[:, 4]), 0.0, 100.0)
        
        # Scale candidate sequence for model input
        cand_np = candidate_seq.detach().numpy()
        cand_scaled = scaler_x.transform(cand_np.reshape(-1, 7)).reshape(1, 4, 7)
        cand_tensor = torch.tensor(cand_scaled, dtype=torch.float32)
        
        pred_reg, _, _ = model(cand_tensor)
        reg_unscaled = scaler_y.inverse_transform(pred_reg.detach().numpy())[0][0]
        pred_score_val = float(reg_unscaled)
        
        # Differentiable loss estimation
        # We approximate differentiable gradient path through scaled tensor
        pred_scaled_target = scaler_y.transform([[target_score]])[0][0]
        loss_target = (pred_reg[0][0] - pred_scaled_target) ** 2
        
        # L2 norm penalty to find MINIMAL recourse distance
        loss_recourse_norm = torch.sum(delta ** 2) * 0.08
        
        loss = loss_target + loss_recourse_norm
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_delta = delta.detach().clone()
            
    # Apply optimal best delta to construct final recourse profile
    final_seq = np.copy(seq_raw)
    final_seq[:, 0] = np.clip(final_seq[:, 0] + np.maximum(0.0, best_delta[:, 0].numpy()), 0.0, 12.0)
    final_seq[:, 1] = np.clip(final_seq[:, 1] + best_delta[:, 1].numpy(), 5.5, 9.0)
    final_seq[:, 2] = np.clip(final_seq[:, 2] + np.maximum(0.0, best_delta[:, 2].numpy()), 0.0, 300.0)
    final_seq[:, 3] = np.clip(final_seq[:, 3] + np.maximum(0.0, best_delta[:, 3].numpy()), 0.0, 10.0)
    final_seq[:, 4] = np.clip(final_seq[:, 4] + np.maximum(0.0, best_delta[:, 4].numpy()), 0.0, 100.0)
    
    # Compute final projected score with optimal recourse features
    final_scaled = scaler_x.transform(final_seq.reshape(-1, 7)).reshape(1, 4, 7)
    with torch.no_grad():
        pred_final, _, _ = model(torch.tensor(final_scaled, dtype=torch.float32))
        achieved_score = float(scaler_y.inverse_transform(pred_final.numpy())[0][0])
        achieved_score = max(0.0, min(100.0, round(achieved_score, 2)))
        
    # Calculate average metric adjustments across weeks for actionable user summary
    study_diff = round(float(np.mean(final_seq[:, 0] - seq_raw[:, 0])), 1)
    sleep_diff = round(float(np.mean(final_seq[:, 1] - seq_raw[:, 1])), 1)
    lms_diff = int(round(float(np.mean(final_seq[:, 2] - seq_raw[:, 2]))))
    assign_diff = round(float(np.mean(final_seq[:, 3] - seq_raw[:, 3])), 1)
    mock_diff = round(float(np.mean(final_seq[:, 4] - seq_raw[:, 4])), 1)
    
    recourse_actions = []
    if study_diff > 0.2:
        recourse_actions.append(f"Increase daily study time by +{study_diff} hrs/day")
    if sleep_diff > 0.2:
        recourse_actions.append(f"Increase sleep & rest recovery by +{sleep_diff} hrs/night")
    elif sleep_diff < -0.3:
        recourse_actions.append(f"Adjust sleep schedule to optimal 7.5 hrs/night")
    if lms_diff > 2:
        recourse_actions.append(f"Increase LMS digital activity by +{lms_diff} logins/week")
    if assign_diff > 0.4:
        recourse_actions.append(f"Complete +{assign_diff} additional assignments per module")
    if mock_diff > 3.0:
        recourse_actions.append(f"Improve mock exam prep by +{mock_diff} marks")
        
    if not recourse_actions:
        recourse_actions.append("Current study parameters are already aligned with target capacity.")
        
    return {
        "target_score": target_score,
        "projected_score": achieved_score,
        "score_gain": round(achieved_score - float(scaler_y.inverse_transform(model(torch.tensor(scaler_x.transform(seq_raw.reshape(-1, 7)).reshape(1, 4, 7), dtype=torch.float32))[0].detach().numpy())[0][0]), 2),
        "recourse_actions": recourse_actions,
        "optimized_metrics": {
            "avg_study_hours": round(float(np.mean(final_seq[:, 0])), 1),
            "avg_sleep_hours": round(float(np.mean(final_seq[:, 1])), 1),
            "avg_lms_logins": int(np.mean(final_seq[:, 2])),
            "avg_assignments_completed": round(float(np.mean(final_seq[:, 3])), 1),
            "avg_mock_exams": round(float(np.mean(final_seq[:, 4])), 1)
        }
    }
