import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

# Action Space definitions for Advisor Agent
ACTIONS = {
    0: {"name": "Rest Mode", "study": -1.0, "sleep": 1.0, "lms": -5.0, "assign": 0.0, "mock": 0.0},
    1: {"name": "Balanced Boost", "study": 1.0, "sleep": 0.0, "lms": 5.0, "assign": 0.0, "mock": 0.0},
    2: {"name": "Intensive Prep", "study": 2.0, "sleep": -1.0, "lms": 10.0, "assign": 0.0, "mock": 0.0},
    3: {"name": "Mock Quiz Drill", "study": 0.5, "sleep": -0.5, "lms": 0.0, "assign": 2.0, "mock": 5.0},
    4: {"name": "Engagement Boost", "study": 0.0, "sleep": 0.5, "lms": 15.0, "assign": 1.0, "mock": 0.0}
}

# Action Space/Behaviors for Student Agent (Compliance/Habits)
STUDENT_BEHAVIORS = {
    0: {"name": "Hyper Study focus", "study": 1.5, "sleep": -1.0, "lms": 5.0, "assign": 0.0, "mock": 2.0},
    1: {"name": "Rest Priority focus", "study": -1.0, "sleep": 1.5, "lms": -5.0, "assign": 0.0, "mock": 0.0},
    2: {"name": "LMS Engagement focus", "study": 0.0, "sleep": 0.0, "lms": 25.0, "assign": 1.0, "mock": 0.0},
    3: {"name": "Assessment Drill focus", "study": 0.5, "sleep": -0.5, "lms": 5.0, "assign": 2.0, "mock": 5.0},
    4: {"name": "Perfect Compliance", "study": 0.0, "sleep": 0.0, "lms": 0.0, "assign": 0.0, "mock": 0.0}
}

class QNetwork(nn.Module):
    """
    Advisor Action Value Q-Network.
    """
    def __init__(self, state_dim=7, action_dim=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class StudentQNetwork(nn.Module):
    """
    Student Action Value Q-Network conditioned on Advisor recommendations.
    """
    def __init__(self, state_dim=7, adv_action_dim=5, action_dim=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + adv_action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
    def forward(self, state, adv_action_onehot):
        x = torch.cat([state, adv_action_onehot], dim=-1)
        return self.network(x)


def transition_marl_state(state, a_advisor, a_student):
    """
    Computes state transition based on both Advisor and Student actions.
    """
    act_adv = ACTIONS.get(a_advisor, ACTIONS[1])
    act_stud = STUDENT_BEHAVIORS.get(a_student, STUDENT_BEHAVIORS[4])
    
    new_state = np.copy(state)
    
    if a_student == 4:
        # Perfect compliance matches Advisor recommendations exactly
        study_delta = act_adv["study"]
        sleep_delta = act_adv["sleep"]
        lms_delta = act_adv["lms"]
        assign_delta = act_adv["assign"]
        mock_delta = act_adv["mock"]
    else:
        # Combined transition matrix representing interactive compliance response
        study_delta = 0.4 * act_adv["study"] + 0.6 * act_stud["study"]
        sleep_delta = 0.4 * act_adv["sleep"] + 0.6 * act_stud["sleep"]
        lms_delta = 0.4 * act_adv["lms"] + 0.6 * act_stud["lms"]
        assign_delta = 0.4 * act_adv["assign"] + 0.6 * act_stud["assign"]
        mock_delta = 0.4 * act_adv["mock"] + 0.6 * act_stud["mock"]
        
    new_state[2] = max(0.0, min(12.0, new_state[2] + study_delta))
    new_state[3] = max(3.0, min(12.0, new_state[3] + sleep_delta))
    new_state[4] = max(0.0, min(300.0, new_state[4] + lms_delta))
    new_state[5] = max(0.0, min(10.0, new_state[5] + assign_delta))
    new_state[6] = max(0.0, min(100.0, new_state[6] + mock_delta))
    
    return new_state


def compute_marl_reward(state_before, state_after, predict_fn):
    """
    Joint cooperative reward balancing grade gains and burnout penalties.
    """
    score_before = predict_fn(state_before)
    score_after = predict_fn(state_after)
    score_gain = score_after - score_before
    
    # Penalize sleep deprivation (burnout indicator)
    sleep_after = state_after[3]
    burnout_penalty = 0.0
    if sleep_after < 6.0:
        burnout_penalty += 2.5 * (6.0 - sleep_after)
        
    # Penalize study fatigue (>10 hours/day study)
    study_after = state_after[2]
    if study_after > 10.0:
        burnout_penalty += 0.5 * (study_after - 10.0)
        
    return score_gain - burnout_penalty


class MARLAgent:
    """
    Cooperative Multi-Agent Reinforcement Learning Wrapper.
    """
    def __init__(self, state_dim=7, action_dim=5, lr=0.005, gamma=0.9):
        self.gamma = gamma
        self.action_dim = action_dim
        
        # Advisor Agent
        self.adv_model = QNetwork(state_dim, action_dim)
        self.adv_optimizer = optim.Adam(self.adv_model.parameters(), lr=lr)
        
        # Student Agent
        self.student_model = StudentQNetwork(state_dim, action_dim, action_dim)
        self.student_optimizer = optim.Adam(self.student_model.parameters(), lr=lr)
        
        self.criterion = nn.MSELoss()
        
    def select_advisor_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.adv_model.eval()
        with torch.no_grad():
            q_vals = self.adv_model(state_tensor)
        return int(torch.argmax(q_vals).item())
        
    def select_student_action(self, state, adv_action, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
            
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        adv_onehot = torch.zeros(1, self.action_dim)
        adv_onehot[0, adv_action] = 1.0
        
        self.student_model.eval()
        with torch.no_grad():
            q_vals = self.student_model(state_tensor, adv_onehot)
        return int(torch.argmax(q_vals).item())

    def train_cooperative(self, transitions):
        if not transitions:
            return
            
        self.adv_model.train()
        self.student_model.train()
        
        # Unpack transitions
        states = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
        adv_actions = torch.tensor([t[1] for t in transitions], dtype=torch.long).unsqueeze(1)
        stud_actions = torch.tensor([t[2] for t in transitions], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([t[3] for t in transitions], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([t[4] for t in transitions]), dtype=torch.float32)
        dones = torch.tensor([t[5] for t in transitions], dtype=torch.float32).unsqueeze(1)
        
        # 1. Train Advisor Q-Network
        q_adv = self.adv_model(states).gather(1, adv_actions)
        with torch.no_grad():
            max_next_q_adv = self.adv_model(next_states).max(1)[0].unsqueeze(1)
            targets_adv = rewards + (1.0 - dones) * self.gamma * max_next_q_adv
            
        loss_adv = self.criterion(q_adv, targets_adv)
        self.adv_optimizer.zero_grad()
        loss_adv.backward()
        self.adv_optimizer.step()
        
        # 2. Train Student Q-Network
        adv_onehot = torch.zeros(len(transitions), self.action_dim)
        adv_onehot.scatter_(1, adv_actions, 1.0)
        
        q_stud = self.student_model(states, adv_onehot).gather(1, stud_actions)
        
        with torch.no_grad():
            next_adv_actions = self.adv_model(next_states).argmax(1).unsqueeze(1)
            next_adv_onehot = torch.zeros(len(transitions), self.action_dim)
            next_adv_onehot.scatter_(1, next_adv_actions, 1.0)
            max_next_q_stud = self.student_model(next_states, next_adv_onehot).max(1)[0].unsqueeze(1)
            targets_stud = rewards + (1.0 - dones) * self.gamma * max_next_q_stud
            
        loss_stud = self.criterion(q_stud, targets_stud)
        self.student_optimizer.zero_grad()
        loss_stud.backward()
        self.student_optimizer.step()

    def save(self, filepath):
        torch.save({
            'adv_model_state_dict': self.adv_model.state_dict(),
            'student_model_state_dict': self.student_model.state_dict()
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self.adv_model.load_state_dict(checkpoint['adv_model_state_dict'])
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.adv_model.eval()
        self.student_model.eval()


def train_rl_advisor(df_all, predict_fn, epochs=150):
    """
    Cooperative Advisor-Student MARL training loops.
    """
    agent = MARLAgent()
    if len(df_all) == 0:
        return agent
        
    cohort_states = []
    for idx, row in df_all.iterrows():
        study = np.mean([float(row.get(f"study_hours_w{w}", 5.0)) for w in range(1, 5)])
        sleep = np.mean([float(row.get(f"sleep_hours_w{w}", 7.5)) for w in range(1, 5)])
        lms = np.mean([float(row.get(f"lms_logins_w{w}", 30.0)) for w in range(1, 5)])
        assign = np.mean([float(row.get(f"assignments_completed_w{w}", 5.0)) for w in range(1, 5)])
        mock = np.mean([float(row.get(f"mock_exams_w{w}", 70.0)) for w in range(1, 5)])
        
        s = np.array([
            float(row.get("attendance", 80.0)),
            float(row.get("previous_marks", 70.0)),
            study, sleep, lms, assign, mock
        ])
        cohort_states.append(s)
        
    epsilon = 0.4
    for epoch in range(epochs):
        transitions = []
        epsilon = max(0.05, epsilon * 0.98)
        
        for s_init in cohort_states:
            s = np.copy(s_init)
            for week in range(4):
                # Cooperative MARL step
                a_adv = agent.select_advisor_action(s, epsilon)
                a_stud = agent.select_student_action(s, a_adv, epsilon)
                
                s_prime = transition_marl_state(s, a_adv, a_stud)
                reward = compute_marl_reward(s, s_prime, predict_fn)
                
                done = 1.0 if week == 3 else 0.0
                transitions.append((s, a_adv, a_stud, reward, s_prime, done))
                s = s_prime
                
        if transitions:
            agent.train_cooperative(transitions)
            
    print(f"[MARL Dynamics Advisor] Cooperative Agent successfully trained for {epochs} steps.")
    return agent
