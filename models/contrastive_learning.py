import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from models.lstm_model import get_seq_and_static_data

class InfoNCELoss(nn.Module):
    """
    Self-Supervised InfoNCE Contrastive Loss.
    Maximizes cosine similarity between augmented representations of the same student profile
    while pushing apart representations from different students.
    """
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        # z_i, z_j shape: [batch_size, dim]
        N = z_i.size(0)
        z_i_norm = nn.functional.normalize(z_i, dim=1)
        z_j_norm = nn.functional.normalize(z_j, dim=1)
        
        representations = torch.cat([z_i_norm, z_j_norm], dim=0)  # shape: [2N, dim]
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature  # shape: [2N, 2N]
        
        # Mask out self-similarity on diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=z_i.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # Targets: z_i[k] corresponds to z_j[k] which is at index N + k
        labels_i = torch.arange(N, 2 * N, device=z_i.device)
        labels_j = torch.arange(0, N, device=z_i.device)
        labels = torch.cat([labels_i, labels_j], dim=0)
        
        loss = self.cross_entropy(similarity_matrix, labels)
        return loss

def augment_sequence_mask(seq_tensor, mask_prob=0.15):
    """
    Data Augmentation 1: Random Feature Masking.
    Randomly zeroes out sequence features to simulate missing LMS logins or partial logs.
    """
    mask = (torch.rand_like(seq_tensor) > mask_prob).float()
    return seq_tensor * mask

def augment_sequence_jitter(seq_tensor, noise_std=0.05):
    """
    Data Augmentation 2: Gaussian Jittering.
    Injects minor continuous variance to continuous parameters.
    """
    noise = torch.randn_like(seq_tensor) * noise_std
    return seq_tensor + noise

def train_contrastive_pretraining(df_all, model, scaler_x, epochs=25, lr=0.003, temperature=0.1):
    """
    Executes Self-Supervised Contrastive Learning (SSL) pre-training on unlabeled sequence logs.
    Regularizes and optimizes shared sequence encoder representations.
    """
    if len(df_all) < 2:
        return False, ["Insufficient student cohort size for contrastive pairs."]
        
    logs = []
    logs.append(f"[SSL Pre-training] Initializing Self-Supervised InfoNCE Contrastive Encoder on {len(df_all)} student logs...")
    
    seq_data, _, _ = get_seq_and_static_data(df_all)
    N, T, F = seq_data.shape
    seq_flat = seq_data.reshape(-1, F)
    seq_scaled = scaler_x.transform(seq_flat).reshape(N, T, F)
    
    base_tensor = torch.tensor(seq_scaled, dtype=torch.float32)
    
    criterion = InfoNCELoss(temperature=temperature)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Create two augmented views of the unlabelled cohort sequences
        view_a = augment_sequence_mask(base_tensor, mask_prob=0.15)
        view_b = augment_sequence_jitter(base_tensor, noise_std=0.05)
        
        # Extract encoder representation representations from model's shared feature map
        # Forward pass returning internal embeddings
        _, _, _ = model(view_a)
        
        # Hook into shared layer activations by projecting shared_fc representations
        z_a = model.shared_fc(model.context_proj(torch.sum(model.transformer_layer(model.lstm(view_a)[0]) * torch.softmax(model.attn_linear(model.transformer_layer(model.lstm(view_a)[0])), dim=1), dim=1)))
        z_b = model.shared_fc(model.context_proj(torch.sum(model.transformer_layer(model.lstm(view_b)[0]) * torch.softmax(model.attn_linear(model.transformer_layer(model.lstm(view_b)[0])), dim=1), dim=1)))
        
        loss = criterion(z_a, z_b)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            logs.append(f"[SSL Pre-training] Epoch {epoch}/{epochs} - InfoNCE Contrastive Loss: {float(loss.item()):.4f}")
            
    model.eval()
    logs.append(f"[SSL Pre-training] Completed self-supervised representation alignment across {N} student profiles.")
    return True, logs
