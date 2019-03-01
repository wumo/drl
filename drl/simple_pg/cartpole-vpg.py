import numpy as np
import gym
from gym.spaces import Discrete, Box
import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from drl.util.dot import make_dot

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

class VanillaNet(nn.Module):
  def __init__(self, output_dim, input_dim, hidden_units=(32,), activation=torch.tanh, output_activation=None):
    super(VanillaNet, self).__init__()
    
    dims = (input_dim,) + hidden_units + (output_dim,)
    self.layers = torch.nn.ModuleList(
      [layer_init(torch.nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
    self.activation = activation
    self.output_activation = output_activation
    self.to(DEVICE)
  
  def forward(self, x):
    x = tensor(x)
    for layer in self.layers[:-1]:
      x = self.activation(layer(x))
    x = self.layers[-1](x)
    if self.output_activation:
      x = self.output_activation(x)
    return x

game = 'CartPole-v0'
env = gym.make(game)

state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n

model = VanillaNet(action_dim, state_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

render = False
lr = 1e-2
epochs = 50
batch_size = 5000

obs = env.reset()
done = False

def reward_to_go(rews):
  n = len(rews)
  rtgs = np.zeros_like(rews)
  for i in reversed(range(n)):
    rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
  return rtgs

# for training policy
def train_one_epoch():
  # make some empty lists for logging.
  batch_obs = []  # for observations
  batch_acts = []  # for actions
  batch_returns = []  # for reward-to-go weighting in policy gradient
  batch_rets = []  # for measuring episode returns
  batch_lens = []  # for measuring episode lengths
  
  # reset episode-specific variables
  obs = env.reset()  # first obs comes from starting distribution
  done = False  # signal from environment that episode is over
  ep_rews = []  # list for rewards accrued throughout ep
  
  finished_rendering_this_epoch = False
  
  # collect experience by acting in the environment with current policy
  while True:
    if (not finished_rendering_this_epoch) and render:
      env.render()
    
    # save obs
    batch_obs.append(obs.copy())
    
    # infer action from policy network
    probs = model(obs)
    m = Categorical(logits=probs)
    action = m.sample().item()
    # act in the environment
    obs, rew, done, _ = env.step(action)
    
    # save action, reward
    batch_acts.append(action)
    ep_rews.append(rew)
    
    if done:
      # if episode is over, record info about episode
      ep_ret, ep_len = sum(ep_rews), len(ep_rews)
      batch_rets.append(ep_ret)
      batch_lens.append(ep_len)
      
      # the weight for each logprob(a_t|s_t) is reward-to-go from t
      batch_returns += list(reward_to_go(ep_rews))
      
      # reset episode-specific variables
      obs, done, ep_rews = env.reset(), False, []
      
      # won't render again this epoch
      finished_rendering_this_epoch = True
      
      # end experience loop if we have enough of it
      if len(batch_obs) > batch_size:
        break
  
  # take a single policy gradient update step
  batch_obs_t = tensor(batch_obs)
  batch_acts_t = tensor(batch_acts)
  batch_returns_t = tensor(batch_returns)
  probs = model(batch_obs_t)
  graph = make_dot(probs)
  graph.view()
  m = Categorical(logits=probs)
  log_prob = m.log_prob(batch_acts_t)
  loss = -log_prob * batch_returns_t
  loss = loss.sum() / len(batch_rets)
  graph = make_dot(probs)
  graph.view()
  batch_loss = loss.item()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return batch_loss, batch_rets, batch_lens

for i in range(epochs):
  batch_loss, batch_rets, batch_lens = train_one_epoch()
  print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
        (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
