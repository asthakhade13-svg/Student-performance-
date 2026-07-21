import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

# Action Space definitions
ACTIONS = {
    0: {"name": "Rest Mode", "study": -1.0, "sleep": 1.0, "lms": -5.0, "assign": 0.0, "mock": 0.0},
    1: {"name": "Balanced Boost", "study": 1.0, "sleep": 0.0, "lms": 5.0, "assign": 0.0, "mock": 0.0},
    2: {"name": "Intensive Prep", "study": 2.0, "sleep": -1.0, "lms": 10.0, "assign": 0.0, "mock": 0.0},
    3: {"name": "Mock Quiz Drill", "study": 0.5, "sleep": -0.5, "lms": 0.0, "assign": 2.0, "mock": 5.0},
    4: {"name": "Engagement Boost", "study": 0.0, "sleep": 0.5, "lms": 15.0, "assign": 1.0, "mock": 0.0}
}

class QNetwork(nn.Module):
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

def transition_state(state, action_idx):
    """
    Applies the action adjustments to the current state vector.
    State layout: [attendance, previous_marks, study_hours, sleep_hours, lms_logins, assignments_completed, mock_exams]
    """
    act = ACTIONS.get(action_idx, ACTIONS[1])
    new_state = np.copy(state)
    
    # Apply modifications
    new_state[2] = max(0.0, min(12.0, new_state[2] + act["study"]))
    new_state[3] = max(3.0, min(12.0, new_state[3] + act["sleep"]))
    new_state[4] = max(0.0, min(300.0, new_state[4] + act["lms"]))
    new_state[5] = max(0.0, min(10.0, new_state[5] + act["assign"]))
    new_state[6] = max(0.0, min(100.0, new_state[6] + act["mock"]))
    
    return new_state

class DQNAgent:
    def __init__(self, state_dim=7, action_dim=5, lr=0.005, gamma=0.9):
        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.action_dim = action_dim
        
    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            q_vals = self.model(state_tensor)
        return int(torch.argmax(q_vals).item())

    def train_on_transitions(self, transitions):
        """
        DQN step update on batch of transitions: (s, a, r, s_prime, done)
        """
        if not transitions:
            return 0.0
        self.model.train()
        
        states = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in transitions], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([t[3] for t in transitions]), dtype=torch.float32)
        dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32).unsqueeze(1)
        
        # Current Q
        q_values = self.model(states).gather(1, actions)
        
        # Max Next Q
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + (1.0 - dones) * self.gamma * max_next_q
            
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

def train_rl_advisor(df_all, predict_fn, epochs=400):
    """
    Offline RL Q-learning training on cohort states.
    `predict_fn` takes a 7-dim state and returns predicted final grade.
    """
    if len(df_all) == 0:
        return DQNAgent()
        
    agent = DQNAgent()
    
    # Construct initial states from database cohort
    cohort_states = []
    for idx, row in df_all.iterrows():
        # Get mean study, sleep, lms, assignments, mock values across weeks
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
        
    # DQN training loop
    epsilon = 0.4
    for epoch in range(epochs):
        transitions = []
        epsilon = max(0.05, epsilon * 0.98)
        
        for s_init in cohort_states:
            s = np.copy(s_init)
            # Run a 4-week trajectory
            for week in range(4):
                a = agent.select_action(s, epsilon)
                s_prime = transition_state(s, a)
                
                # Compute rewards based on grade difference
                score_before = predict_fn(s)
                score_after = predict_fn(s_prime)
                reward = score_after - score_before
                
                done = 1.0 if week == 3 else 0.0
                transitions.append((s, a, reward, s_prime, done))
                s = s_prime
                
        # Optimize DQN
        if transitions:
            agent.train_on_transitions(transitions)
            
    print(f"[RL Policy Network] Offline DQN Agent successfully trained for {epochs} steps.")
    return agent
