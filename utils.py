import numpy as np
from collections import deque
import random


class OUNoise(object):   # action refers to action_space which contains all information of action in the environment
    def __init__(self, action_space, mu=0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay=10000):   # action_space and dimension need modification
        self.mu = mu
        self.sigma = max_sigma
        self.decay = decay
        self.theta = theta
        self.action_dim = action_space.shape[1] # row length
        self.low = action_space.low
        self.high = action_space.high
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t = 0):
        ou_state = self.evolve_state
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay)
        return np.clip(action + ou_state, self.low, self.high)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.memory, batch_size))
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        next_state = np.array(next_state)
        return state, action, reward, next_state

    def __len__(self):
        return len(self.memory)


def Adjacency(data, i): # data refers to neighboring vehicles vector to certain vehicle
    adj_dim = np.count_nonzero(data[i, :]) # i needs revision!!!!!!
    nonzero_index = np.nonzero(data[i, :])
    nonzero_index = np.delete(nonzero_index, int(np.argwhere(nonzero_index == i)))
    adj_matrix = np.zeros((adj_dim, data.shape[1]))
    adj_matrix[0][i] = 1 # refer to the agent itself
    for j in range(1, adj_matrix.shape[0]):
        adj_matrix[j][nonzero_index[j]] = 1
    return adj_matrix