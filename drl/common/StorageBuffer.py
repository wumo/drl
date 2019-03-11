import torch

class StorageBuffer:
    def __init__(self, size, keys=None):
        self.keys = keys
        self.size = size
        self.reset()
    
    def reset(self):
        for key in self.keys:
            setattr(self, key, [])
    
    def placeholder(self):
        for key in self.keys:
            v = getattr(self, key)
            if len(v) == 0:
                setattr(self, key, [None] * self.size)
    
    def append(self, **kwargs):
        for key, value in kwargs.items():
            assert key in self.keys
            getattr(self, key).append(value)
    
    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
