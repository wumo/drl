from drl.algo.BaseAgent import BaseAgent
from drl.util.torch_utils import toNumpy, range_tensor, toTensor, epsilon_greedy, huber_loss
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

# Human-level control through deep reinforcement (DQN)
class QuantileRegressionDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        
        self.replay = config.replay_fn()
        
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        
        self.task = config.task_fn()
        self.states = self.task.reset()
        
        self.batch_indices = range_tensor(self.config.batch_size)
        
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = toTensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)
    
    def step(self):
        config = self.config
        
        # rollout
        for _ in range(self.config.rollout_length):
            # choose according to max(Q)
            q = self.network(config.state_normalizer(self.states)).mean(-1)
            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, toNumpy(q))
            
            next_states, rewards, dones, infos = self.task.step(actions)
            state, reward, next_state, done, info = self.states[0], rewards[0], next_states[0], int(dones[0]), infos[0]
            self.states = next_states
            self.total_steps += 1
            
            reward = config.reward_normalizer(reward)
            self.replay.store([state, actions[0], reward, next_state, done])
        
        if self.total_steps > config.exploration_steps:
            # minibatch gradient descent
            experiences = self.replay.sample(config.batch_size)
            states, actions, rewards, next_states, terminals = experiences
            states = config.state_normalizer(states)
            next_states = config.state_normalizer(next_states)
            
            quantiles_next = self.target_network(next_states).detach()
            a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
            quantiles_next = quantiles_next[self.batch_indices, a_next, :]
            
            rewards = toTensor(rewards).unsqueeze(-1)
            terminals = toTensor(terminals).unsqueeze(-1)
            quantiles_next = rewards + self.config.discount * (1 - terminals) * quantiles_next
            
            quantiles = self.network(states)
            actions = toTensor(actions).long()
            quantiles = quantiles[self.batch_indices, actions, :]
            
            quantiles_next = quantiles_next.t().unsqueeze(-1)
            diff = quantiles_next - quantiles
            loss = huber_loss(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs()
            
            self.optimizer.zero_grad()
            loss.mean(0).mean(1).sum().backward()
            clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        
        if self.total_steps / config.rollout_length % config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
