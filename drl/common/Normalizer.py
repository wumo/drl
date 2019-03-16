import numpy as np

class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only
    
    def set_read_only(self):
        self.read_only = True
    
    def unset_read_only(self):
        self.read_only = False
    
    def state_dict(self):
        return None
    
    def load_state_dict(self, _):
        return

class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef
    
    def __call__(self, x):
        x = np.asarray(x)
        return self.coef * x

class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        super().__init__(1.0 / 255)

class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)
