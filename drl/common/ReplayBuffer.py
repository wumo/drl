import numpy as np
from drl.util.utils import combined_shape

class ReplayBuffer:
    def __init__(self, env, memory_size, batch_size, stack=1):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.state_shape = env.observation_space.shape
        self.state_dtype = env.observation_space.dtype
        self.action_shape = env.action_space.shape
        self.action_dtype = env.action_space.dtype
        self.stack = stack
        if self.stack > 1: self.state_shape = self.state_shape[1:]
        self.states = np.zeros(combined_shape(memory_size + self.stack - 1, self.state_shape), dtype=self.state_dtype)
        self.next_states = np.zeros(combined_shape(memory_size + self.stack - 1, self.state_shape),
                                    dtype=self.state_dtype)
        self.actions = np.zeros(combined_shape(memory_size, self.action_shape), dtype=self.action_dtype)
        self.rewards = np.zeros(memory_size, dtype=np.float32)
        self.dones = np.zeros(memory_size, dtype=np.int)
        self.state_ptr = self.stack - 1
        self.ptr = 0
        self.size = 0
    
    def store(self, experience):
        state, action, reward, next_state, done = experience
        # if stacked, only store the last frame)
        state = state if self.stack == 1 else state[-1]
        next_state = next_state if self.stack == 1 else next_state[-1]
        self.states[self.state_ptr] = state
        self.next_states[self.state_ptr] = next_state
        
        if self.size == 0 and self.stack > 1:  # stack first (self.stack-1) elements
            for i in range(self.state_ptr):
                self.states[i] = state
                self.next_states[i] = next_state
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.state_ptr = (self.state_ptr + 1) % (self.memory_size + self.stack - 1)
        self.ptr = (self.ptr + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)
    
    def store_batch(self, experiences):
        for experience in experiences:
            self.store(experience)
    
    def sample(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        sampled_indices = np.random.randint(0, self.size, size=batch_size)
        sampled_actions = self.actions[sampled_indices]
        sampled_rewards = self.rewards[sampled_indices]
        sampled_dones = self.dones[sampled_indices]
        if self.stack == 1:
            sampled_states = self.states[sampled_indices]
            sampled_next_states = self.next_states[sampled_indices]
        else:
            zero_indice = self.stack - 1 if self.size < self.memory_size \
                else (self.state_ptr + self.stack - 1) % (self.memory_size + self.stack - 1)
            
            def stack(states):
                sampled_states = np.empty(combined_shape(batch_size, (self.stack,) + self.state_shape),
                                          dtype=self.state_dtype)
                for i, indice in enumerate(sampled_indices):
                    indice = (indice + zero_indice) % (self.memory_size + self.stack - 1)
                    if indice < self.stack - 1:
                        end = self.stack - indice - 1
                        sampled_states[i] = np.concatenate([states[-end:], states[:indice + 1]], axis=0)
                    else:
                        sampled_states[i] = np.concatenate([states[indice - self.stack + 1:indice + 1]], axis=0)
                return sampled_states
            
            sampled_states = stack(self.states)
            sampled_next_states = stack(self.next_states)
        
        return [sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_dones]
