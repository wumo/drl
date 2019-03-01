import torch
import torch.nn  as nn
from math import log10, floor

def round_sig(x, sig=4):
    x = float(x)
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

DEVICE = torch.device('cuda:0')

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
    return x

def tensors(x):
    return [tensor(i) for i in x]
