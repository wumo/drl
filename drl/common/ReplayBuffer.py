import numpy as np
from drl.util.utils import combined_shape

class ReplayBuffer:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = [None] * memory_size
        self.size = 0
        self.pos = 0
    
    def store(self, experience):
        self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)
    
    def store_batch(self, experiences):
        for experience in experiences:
            self.store(experience)
    
    def sample(self, batch_size=None):
        if self.size == 0: return None
        if batch_size is None: batch_size = self.batch_size
        
        sampled_indices = np.random.randint(0, self.size, size=batch_size)
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.array(x,copy=False), zip(*sampled_data)))
        return batch_data
