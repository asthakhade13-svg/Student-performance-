import os
import json
import torch

PERSONALIZED_HEADS_FILE = os.path.join("models", "personalized_heads.json")

def load_personalized_heads():
    if not os.path.exists(PERSONALIZED_HEADS_FILE):
        return {}
    try:
        with open(PERSONALIZED_HEADS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_personalized_heads(data):
    try:
        with open(PERSONALIZED_HEADS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print("Error saving personalized heads:", e)

def apply_personalization(model, student_id):
    heads = load_personalized_heads()
    if student_id in heads:
        student_data = heads[student_id]
        weight_list = student_data.get("reg_head.weight")
        bias_list = student_data.get("reg_head.bias")
        
        if weight_list and bias_list:
            with torch.no_grad():
                model.reg_head.weight.copy_(torch.tensor(weight_list, dtype=torch.float32))
                model.reg_head.bias.copy_(torch.tensor(bias_list, dtype=torch.float32))
            print(f"[Personalization] Loaded and applied custom neural head parameters for '{student_id}'")
            return True
    return False

def train_personalized_head(model, seq_x, actual_score, scaler_y, student_id):
    # Unfreeze only the regression head parameters for fine-tuning
    for param in model.parameters():
        param.requires_grad = False
    for param in model.reg_head.parameters():
        param.requires_grad = True
        
    # Standardize target actual_score using scaler_y
    target_scaled = scaler_y.transform([[actual_score]])[0][0]
    target_tensor = torch.tensor([[target_scaled]], dtype=torch.float32)
    input_tensor = torch.tensor(seq_x, dtype=torch.float32)
    
    # Optimizer for reg_head only
    optimizer = torch.optim.Adam(model.reg_head.parameters(), lr=0.002)
    criterion = torch.nn.MSELoss()
    
    model.train()
    # 15 epochs of online gradient descent step tuning
    for epoch in range(15):
        optimizer.zero_grad()
        pred_reg, _, _ = model(input_tensor)
        loss = criterion(pred_reg, target_tensor)
        loss.backward()
        optimizer.step()
        
    # Save the updated parameters to personalized_heads.json
    heads = load_personalized_heads()
    heads[student_id] = {
        "reg_head.weight": model.reg_head.weight.detach().numpy().tolist(),
        "reg_head.bias": model.reg_head.bias.detach().numpy().tolist()
    }
    save_personalized_heads(heads)
    print(f"[Personalization] Fine-tuned and saved personalized layer for '{student_id}'. Loss: {float(loss):.4f}")
