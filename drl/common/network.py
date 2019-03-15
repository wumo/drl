from gym.spaces import Box, Discrete
import torch
import torch.nn  as nn
import torch.nn.functional as F
from drl.util.utils import DEVICE, toTensor
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim
    
    def forward(self, x):
        return x

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(32,), activation=torch.tanh):
        super(FCBody, self).__init__()
        
        dims = list(state_dim) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.activation = activation
        self.feature_dim = dims[-1]
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, shared_body=None, actor_body=None, critic_body=None):
        super(ActorCriticNet, self).__init__()
        if shared_body is None: shared_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(shared_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(shared_body.feature_dim)
        self.shared_body = shared_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim))
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1))
        
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.shared_params = list(self.shared_body.parameters())
    
    def forward(self, obs):
        obs = toTensor(obs)
        shared = self.shared_body(obs)
        
        action, log_prob = self.actor(obs, None, shared)
        v = self.critic(obs, shared)
        return action, v, log_prob
    
    def actor(self, obs, action, shared=None):
        raise NotImplementedError
    
    def critic(self, obs, shared=None):
        if shared is None: shared = self.shared_body(toTensor(obs))
        return self.fc_critic(self.critic_body(shared))

class CategoricalActorCriticNet(ActorCriticNet):
    def __init__(self, state_dim, action_dim, shared_body=None, actor_body=None, critic_body=None):
        super(CategoricalActorCriticNet, self).__init__(state_dim, action_dim, shared_body, actor_body, critic_body)
        self.to(DEVICE)
    
    def actor(self, obs, action=None, shared=None):
        if shared is None: shared = self.shared_body(toTensor(obs))
        action_logits = self.fc_action(self.actor_body(shared))
        action_dist = Categorical(logits=action_logits)
        if action is None:
            action = action_dist.sample()
            return action, action_dist.log_prob(action)
        else:
            return action_dist.log_prob(action)

class GaussianActorCriticNet(ActorCriticNet):
    def __init__(self, state_dim, action_dim, shared_body=None, actor_body=None, critic_body=None):
        super(GaussianActorCriticNet, self).__init__(state_dim, action_dim, shared_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.critic_params.append(self.std)
        self.to(DEVICE)
    
    def actor(self, obs, action=None, shared=None):
        if shared is None: shared = self.shared_body(toTensor(obs))
        action_mean = self.fc_action(self.actor_body(shared))
        action_dist = Normal(action_mean, F.softplus(self.std))
        if action is None:
            action = action_dist.sample()
            return action, action_dist.log_prob(action).sum(-1)
        else:
            return action_dist.log_prob(action).sum(-1)

def mlp_actor_critic(observation_space, action_space, hidden_sizes=(32,), activation=torch.tanh):
    # assert isinstance(action_space, gym.spaces.discrete.Discrete), "Currently only support Discrete."
    state_dim = observation_space.shape
    if isinstance(action_space, Box):
        action_dim = action_space.shape[0]
        net = GaussianActorCriticNet(state_dim, action_dim,
                                     actor_body=FCBody(state_dim, hidden_sizes, activation),
                                     critic_body=FCBody(state_dim, hidden_sizes, activation))
    elif isinstance(action_space, Discrete):
        action_dim = action_space.n
        net = CategoricalActorCriticNet(state_dim, action_dim,
                                        actor_body=FCBody(state_dim, hidden_sizes, activation),
                                        critic_body=FCBody(state_dim, hidden_sizes, activation))
    else:
        assert False, "Error Space"
    return net
