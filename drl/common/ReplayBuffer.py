import numpy as np
from drl.util.utils import combined_shape

class ReplayBuffer:
    
    def __init__(self, state_dim, action_dim, memory_size, batch_size):
        self.states = np.zeros([memory_size, state_dim], dtype=np.float32)
        self.next_states = np.zeros([memory_size, state_dim], dtype=np.float32)
        self.actions = np.zeros(combined_shape(memory_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(memory_size, dtype=np.float32)
        self.dones = np.zeros(memory_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
    
    def store(self, experience):
        state, action, reward, next_state, done = experience
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)
    
    def store_batch(self, experiences):
        for experience in experiences:
            self.store(experience)
    
    def sample(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        sampled_indices = np.random.randint(0, self.size, size=batch_size)
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.dones[sampled_indices]]
