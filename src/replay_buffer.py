import numpy as np
import torch
import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)


    def add(self, state, action, reward, next_state, done):

        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size):
        """we need to turn 
            batch = [(State_1, Action_1, Reward_1, NextState_1, Done_1),  # Memory 1
                    (State_2, Action_2, Reward_2, NextState_2, Done_2),  # Memory 2
                    (State_3, Action_3, Reward_3, NextState_3, Done_3)   # Memory 3]    
            to
            states      = (State_1, State_2, State_3)
            actions     = (Action_1, Action_2, Action_3)
            rewards     = (Reward_1, Reward_2, Reward_3)
            next_states = (NextState_1, NextState_2, NextState_3)
            dones       = (Done_1, Done_2, Done_3)"""
        
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )


    def __len__(self):
        return len(self.buffer)