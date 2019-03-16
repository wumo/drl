import numpy as np
from drl.util.utils import combined_shape

class ReplayBuffer:
    def __init__(self, memory_size, batch_size, stack=1):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.stack = stack
        self.states = [None] * memory_size
        self.next_states = [None] * memory_size
        self.actions = [None] * memory_size
        self.rewards = [None] * memory_size
        self.dones = [None] * memory_size
        self.ptr = 0
        self.size = 0
        
        # self.memory_size = memory_size
        # self.batch_size = batch_size
        # self.data = [None] * memory_size
        # self.size = 0
        # self.pos = 0
    
    def store(self, experience):
        state, action, reward, next_state, done = experience
        # if stacked, only store the last frame)
        self.states[self.ptr] = state if self.stack == 1 else np.expand_dims(state[-1], axis=0)
        self.next_states[self.ptr] = next_state if self.stack == 1 else np.expand_dims(next_state[-1], axis=0)
        # np.expand_dims(next_state[-1], axis=0)
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
        actions = np.asarray([self.actions[index] for index in sampled_indices])
        rewards = np.asarray([self.rewards[index] for index in sampled_indices])
        dones = np.asarray([self.dones[index] for index in sampled_indices])
        if self.stack == 1:
            states = np.asarray([self.states[index] for index in sampled_indices])
            next_states = np.asarray([self.next_states[index] for index in sampled_indices])
        else:
            def stack(states):
                sampled_states = [None] * batch_size
                for i, indice in enumerate(sampled_indices):
                    if indice < self.stack - 1:
                        if self.size < self.memory_size:
                            state = states[indice]
                            sampled_states[i] = np.concatenate([state] * self.stack, axis=0)
                        else:  # full cycle buffer
                            end = self.stack - indice - 1
                            sampled_states[i] = np.concatenate(states[-end:] + states[:indice + 1], axis=0)
                    else:
                        sampled_states[i] = np.concatenate(states[indice - self.stack + 1:indice + 1], axis=0)
                return sampled_states
            
            states = np.asarray(stack(self.states))
            next_states = np.asarray(stack(self.next_states))
        
        return [states, actions, rewards, next_states, dones]
    
    # def store(self, state, action, reward, next_state, terminal):
    #     if self.stack > 1:
    #         state = np.expand_dims(state[-1], axis=0)
    #         next_state = np.expand_dims(state[-1], axis=0)
    #     self.data[self.pos] = [state, action, reward, next_state, terminal]
    #     self.pos = (self.pos + 1) % self.memory_size
    #     self.size = min(self.size + 1, self.memory_size)
    #
    # def store_batch(self, experiences):
    #     for experience in experiences:
    #         self.store(experience)
    #
    # def sample(self, batch_size=None):
    #     if self.size == 0: return None
    #     if batch_size is None: batch_size = self.batch_size
    #
    #     sampled_indices = np.random.randint(0, self.size, size=batch_size)
    #     sampled_data = [self.data[ind] for ind in sampled_indices]
    #     batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
    #     return batch_data
