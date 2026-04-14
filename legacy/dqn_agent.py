import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# =====================================================
# Q NETWORK
# =====================================================
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# DQN AGENT (FIXED + STABLE)
# =====================================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = deque(maxlen=20000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.lr = 1e-3

        self.model = QNet(state_dim, action_dim)
        self.target = QNet(state_dim, action_dim)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.loss_fn = nn.MSELoss()

    # ----------------------------------------
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            q_values = self.model(state)

        return torch.argmax(q_values).item()

    # ----------------------------------------
    def remember(self, s, a, r, s2, done=False):
        self.memory.append((s, a, r, s2, done))

    # ----------------------------------------
    def train(self, batch_size=64):

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # current Q
        q_values = self.model(states).gather(1, actions)

        # next Q
        with torch.no_grad():
            max_next_q = self.target(next_states).max(1, keepdim=True)[0]

        target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ----------------------------------------
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())