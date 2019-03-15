#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from drl.common.DeviceSetting import DEVICE
from .network_bodies import *
from drl.util.utils import toTensor
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(DEVICE)
    
    def forward(self, x):
        phi = self.body(toTensor(x))
        y = self.fc_head(phi)
        return y

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(DEVICE)
    
    def forward(self, x):
        phi = self.body(toTensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(DEVICE)
    
    def forward(self, x):
        phi = self.body(toTensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(DEVICE)
    
    def forward(self, x):
        phi = self.body(toTensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(DEVICE)
    
    def forward(self, x):
        phi = self.body(toTensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, shared_body, actor_body, critic_body):
        super().__init__()
        if shared_body is None: shared_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(shared_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(shared_body.feature_dim)
        self.shared_body = shared_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.shared_params = list(self.shared_body.parameters())
    
    def forward(self, obs, action=None):
        shared = self.feature(obs)
        
        action, log_prob, entropy = self.actor(obs, action, shared)
        v = self.critic(obs, shared)
        return action, log_prob, entropy, v
    
    def feature(self, obs):
        obs = toTensor(obs)
        return self.shared_body(obs)
    
    def actor(self, obs, action, shared=None) -> list:
        raise NotImplementedError
    
    def critic(self, obs, shared=None):
        if shared is None: shared = self.shared_body(toTensor(obs))
        return self.fc_critic(self.critic_body(shared))

class DeterministicActorCriticNet(ActorCriticNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 shared_body=None,
                 actor_body=None,
                 critic_body=None):
        super().__init__(state_dim, action_dim, shared_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.actor_params + self.network.shared_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.network.shared_params)
        self.to(DEVICE)
    
    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action
    
    def feature(self, obs):
        obs = toTensor(obs)
        return self.shared_body(obs)
    
    def actor(self, phi):
        return F.tanh(self.fc_action(self.network.actor_body(phi)))
    
    def critic(self, phi, a):
        return self.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(ActorCriticNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 shared_body=None,
                 actor_body=None,
                 critic_body=None):
        super().__init__(state_dim, action_dim, shared_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.actor_params.append(self.std)
        self.to(DEVICE)
    
    def actor(self, obs, action=None, shared=None):
        if shared is None: shared = self.feature(obs)
        # action_mean = self.fc_action(self.actor_body(shared))
        action_mean = F.tanh(self.fc_action(self.actor_body(shared)))
        action_dist = Normal(action_mean, F.softplus(self.std))
        if action is None:
            action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = action_dist.entropy().sum(-1).unsqueeze(-1)
        return action, log_prob, entropy

class CategoricalActorCriticNet(ActorCriticNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 shared_body=None,
                 actor_body=None,
                 critic_body=None):
        super().__init__(state_dim, action_dim, shared_body, actor_body, critic_body)
        self.to(DEVICE)
    
    def actor(self, obs, action=None, shared=None):
        if shared is None: shared = self.feature(obs)
        action_logits = self.fc_action(self.actor_body(shared))
        action_dist = Categorical(logits=action_logits)
        if action is None:
            action = action_dist.sample()
        log_prob = action_dist.log_prob(action).unsqueeze(-1)
        entropy = action_dist.entropy().unsqueeze(-1)
        return action, log_prob, entropy
