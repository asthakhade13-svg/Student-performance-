import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sqlite3

SKILLS = ["Algebra", "Calculus", "Mechanics"]
NUM_SKILLS = len(SKILLS)

class StudentDKT(nn.Module):
    """
    Deep Knowledge Tracing (DKT) PyTorch Model.
    Traces skill mastery trajectories over question interaction sequences.
    """
    def __init__(self, num_skills=3, embed_dim=16, hidden_dim=32):
        super(StudentDKT, self).__init__()
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        
        # Embed skill indices: shape [num_skills, embed_dim]
        self.skill_embed = nn.Embedding(num_skills, embed_dim)
        
        # Input size: embed_dim (skill representation) + 1 (binary correct/incorrect indicator)
        self.lstm = nn.LSTM(input_size=embed_dim + 1, hidden_size=hidden_dim, batch_first=True)
        
        # Output layer maps to mastery probability for each skill
        self.out_layer = nn.Linear(hidden_dim, num_skills)
        
    def forward(self, skill_indices, correctness, hidden=None):
        # skill_indices: [batch_size, seq_len]
        # correctness: [batch_size, seq_len]
        batch_size, seq_len = skill_indices.size()
        
        embeds = self.skill_embed(skill_indices)  # [batch_size, seq_len, embed_dim]
        inputs = torch.cat([embeds, correctness.unsqueeze(-1)], dim=-1)  # [batch_size, seq_len, embed_dim + 1]
        
        lstm_out, hidden = self.lstm(inputs, hidden)  # [batch_size, seq_len, hidden_dim]
        
        logits = self.out_layer(lstm_out)  # [batch_size, seq_len, num_skills]
        probs = torch.sigmoid(logits)
        
        return probs, hidden

def load_student_interactions(student_id, db_path=None):
    if db_path is None:
        db_path = os.path.join("models", "student_records.db")
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT skill_id, is_correct FROM student_quiz_logs 
        WHERE student_id = ? 
        ORDER BY week ASC
    """, (student_id,))
    rows = c.fetchall()
    conn.close()
    
    skill_seq = []
    correct_seq = []
    for s_id, correct in rows:
        if s_id in SKILLS:
            skill_seq.append(SKILLS.index(s_id))
            correct_seq.append(float(correct))
            
    return skill_seq, correct_seq

def get_student_mastery(model, student_id, db_path=None):
    """
    Feeds the historical interaction logs to the DKT model to predict 
    the student's current probability of mastery for Algebra, Calculus, and Mechanics.
    """
    model.eval()
    skills_seq, correct_seq = load_student_interactions(student_id, db_path)
    
    # Default baseline mastery probabilities if no history exists (Zero-shot DKT)
    default_mastery = {"Algebra": 0.60, "Calculus": 0.55, "Mechanics": 0.58}
    
    if not skills_seq:
        return default_mastery
        
    skill_tensor = torch.tensor([skills_seq], dtype=torch.long)
    correct_tensor = torch.tensor([correct_seq], dtype=torch.float32)
    
    with torch.no_grad():
        probs, _ = model(skill_tensor, correct_tensor)
        # Select predictions at final sequence step: shape [num_skills]
        final_probs = probs[0, -1].numpy()
        
    mastery = {}
    for idx, skill in enumerate(SKILLS):
        mastery[skill] = round(float(final_probs[idx]), 3)
        
    return mastery

def train_dkt_model(model, db_path=None, epochs=50, lr=0.01):
    """
    Trains the DKT model on quiz interactions of all students.
    Uses next-step validation prediction to compute binary cross entropy loss.
    """
    if db_path is None:
        db_path = os.path.join("models", "student_records.db")
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT DISTINCT student_id FROM student_quiz_logs")
    students = [row[0] for row in c.fetchall()]
    conn.close()
    
    if len(students) < 2:
        return
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        student_count = 0
        
        for s_id in students:
            skills_seq, correct_seq = load_student_interactions(s_id, db_path)
            if len(skills_seq) < 2:
                continue
                
            skill_tensor = torch.tensor([skills_seq[:-1]], dtype=torch.long)
            correct_tensor = torch.tensor([correct_seq[:-1]], dtype=torch.float32)
            
            # Targets: the correctness of the next quiz step
            # target_skills indicates which skill correctness is verified next
            target_skills = skills_seq[1:]
            target_correctness = correct_seq[1:]
            
            optimizer.zero_grad()
            probs, _ = model(skill_tensor, correct_tensor) # shape: [1, seq_len - 1, num_skills]
            
            # Extract predicted probabilities corresponding to next-step skill categories
            seq_len = len(target_skills)
            preds = torch.zeros(seq_len)
            for t in range(seq_len):
                preds[t] = probs[0, t, target_skills[t]]
                
            targets = torch.tensor(target_correctness, dtype=torch.float32)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            student_count += 1
            
        if student_count == 0:
            break
            
    print(f"[DKT Training] Completed {epochs} epochs. Average Loss: {epoch_loss / max(1, student_count):.4f}")
