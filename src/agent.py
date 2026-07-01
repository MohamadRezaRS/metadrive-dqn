import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.network import DQNNetwork

class DQNAgent:
    def __init__(self, state_size,hidden_size, action_size, lr=3e-4, gamma=0.99 , device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.hidden_size = hidden_size
        self.device=device

        # 1. two networks: a primary Q-network and a Target network
        self.policy_net = DQNNetwork(state_size, hidden_size , action_size=action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, hidden_size , action_size=action_size).to(self.device)

        # copy the exact starting weights from policy to target
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 

        # 2. the optimizer (the gradient descent math)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state, epsilon=0.1):
        """
        E-greedy action selection
        """
        if random.random() < epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        return torch.argmax(q_values).item()

    def learn(self, experiences):
        """
        applies the Bellman update and trains the neural network
        """
        states, actions, rewards, next_states, dones = experiences


        # Push the entire batch of experiences to the GPU
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        

        # 1. calculate current Q(s, a)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. calculate target y: y = r + (1 - d) * gamma * max(Q_target(s', a'))
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 3. calculate MSE loss : L = (Q(s,a) - y)^2
        loss = self.loss_fn(current_q, target_q)

        # 4. backpropagation
        self.optimizer.zero_grad() 
        loss.backward()            
        self.optimizer.step()      

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())