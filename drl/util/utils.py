import torch
import torch.nn  as nn
import numpy as np
from math import log10, floor

DEVICE = torch.device('cuda:0')

def round_sig(x, sig=4):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x .
  
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
  
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n
    
    global_sum_sq = np.sum((x - mean) ** 2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std
    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, global_min, global_max
    return mean, std

def toTensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
    return x

def toTensors(x):
    return [toTensor(i) for i in x]
