import numpy as np
import scipy.signal

EPS = 1e-8

def combined_shape(length, shape=None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
  """
  magic from rllab for computing discounted cumulative sums of vectors.

  input:
      vector x,
      [x0,
       x1,
       x2]

  output:
      [x0 + discount * x1 + discount^2 * x2,
       x1 + discount * x2,
       x2]
  """
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def statistics_scalar(x):
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
  return mean, std
