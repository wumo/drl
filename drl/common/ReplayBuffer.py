import numpy as np
from drl.util.utils import combined_shape
from collections import deque

class ReplayBuffer:
    def __init__(self, env, memory_size, stack=1):
        self.memory_size = memory_size
        self.state_shape = env.observation_space.shape
        self.state_dtype = env.observation_space.dtype
        self.action_shape = env.action_space.shape
        self.action_dtype = env.action_space.dtype
        self.stack = stack
        if self.stack > 1: self.state_shape = self.state_shape[1:]
        # np.ones() preallocates the array's memory while np.zeros() doesn't
        self.states = np.ones(combined_shape(memory_size, self.state_shape), self.state_dtype)
        self.memo = deque([], maxlen=self.stack - 1)
        
        self.actions = np.ones(combined_shape(memory_size, self.action_shape), self.action_dtype)
        self.rewards = np.zeros(memory_size, dtype=np.float32)
        self.terminals = np.zeros(memory_size, dtype=np.int)
        self.ptr = 0
        self.size = 0
    
    def index(self, index):
        return (index + self.memory_size) % self.memory_size
    
    def increment(self):
        self.ptr = (self.ptr + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)
    
    def isStacked(self):
        return self.stack > 1
    
    def isNotStacked(self):
        return not self.isStacked()
    
    def isEmpty(self):
        return self.size == 0
    
    def isFull(self):
        return self.size >= self.memory_size
    
    def memo_state(self, state):
        self.memo.append(state)  # current state becomes history
    
    def memo_start_state(self, start_state):
        for _ in range(self.stack):
            self.memo.append(start_state)
    
    def stack_store(self, state):
        # if stacked, only store the last frame
        state = state[-1]
        if self.isFull():  # ptr overlap heading elements, (self.ptr+1) will be the first state.
            self.memo_state(self.states[self.ptr])  # current state becomes history
            first_index = (self.ptr + 1) % self.memory_size
            if self.terminals[first_index]:
                # first state is start state, memo will only contain it.
                self.memo_start_state(self.states[first_index])
        elif self.isEmpty():
            self.memo_start_state(state)
        self.states[self.ptr] = state
    
    def store(self, experience):
        state, action, reward, next_state, terminal = experience
        
        if self.isEmpty():
            self.terminals[self.ptr] = 1  # first element is terminal(start state)
            if self.isStacked():
                self.stack_store(state)
            else:
                self.states[self.ptr] = state
            self.increment()
        
        # state and next_state are consecutive, so we only store current one
        if self.isStacked():
            self.stack_store(next_state)
        else:
            self.states[self.ptr] = next_state
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminals[self.ptr] = terminal
        self.increment()
    
    def history_concat(self, indices):
        states = np.zeros(combined_shape(indices.shape[0], self.stack, self.state_shape), dtype=self.state_dtype)
        for i, index in enumerate(indices):
            memo_index = self.memo.maxlen
            for j in reversed(range(self.stack)):
                if index == self.ptr:  # first element, first from itself, then from memo
                    states[i][j] = self.memo[memo_index] if memo_index < self.memo.maxlen else self.states[index]
                    memo_index -= 1
                elif self.terminals[index]:
                    states[i][j] = self.states[index]
                else:
                    states[i][j] = self.states[index]
                    index = (index - 1 + self.memory_size) % self.memory_size
        return states
    
    def sample(self, batch_size):
        sampled_next_indices = np.random.randint(1, self.size, size=batch_size)
        if self.isFull(): sampled_next_indices = (sampled_next_indices + self.ptr) % self.memory_size
        sampled_indices = (sampled_next_indices - 1 + self.memory_size) % self.memory_size
        
        sampled_actions = self.actions[sampled_next_indices]
        sampled_rewards = self.rewards[sampled_next_indices]
        sampled_terminals = self.terminals[sampled_next_indices]
        if self.isNotStacked():
            sampled_states = self.states[sampled_indices]
            sampled_next_states = self.states[sampled_next_indices]
        else:
            sampled_states = self.history_concat(sampled_indices)
            sampled_next_states = self.history_concat(sampled_next_indices)
        
        return [sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminals]
