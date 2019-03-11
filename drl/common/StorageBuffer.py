import torch

class StorageBuffer:
    def __int__(self, size):
        self.size = size
        self.ptr = 0
    
    def store_next(self, **kwargs):
        assert self.ptr < self.size
        for key, value in kwargs.items():
            getattr(self, key)[self.ptr] = value
        self.ptr += 1
    
    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
