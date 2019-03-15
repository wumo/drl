from drl.algo.BaseAgent import BaseAgent
from drl.common.StorageBuffer import StorageBuffer
from drl.util.torch_utils import toNumpy, range_tensor, toTensor
from drl.util.misc import random_sample
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

class PPO2StorageBuffer(StorageBuffer):
    def __init__(self, size):
        super().__int__(size)
        self.states = [None] * size
        self.actions = [None] * size
        self.rewards = [None] * size
        self.values = [None] * size
        self.log_pi = [None] * size
        self.entropy = [None] * size
        self.terminals = [None] * size
        self.advantages = [None] * size
        self.returns = [None] * size

class PPO2Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.network = config.network_fn()
        self.act_optimizer = config.act_optimizer_fn(self.network.actor_params + self.network.shared_params)
        self.critic_optimizer = config.critic_optimizer_fn(self.network.critic_params + self.network.shared_params)
        
        self.task = config.task_fn()
        self.states = config.state_normalizer(self.task.reset())
        
        self.online_rewards = np.zeros(config.num_workers)
    
    def step(self):
        config = self.config
        storage = PPO2StorageBuffer(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            action_tr, log_prob_tr, entropy_tr, v_tr = self.network(states)
            next_states, rewards, terminals, infos = self.task.step(toNumpy(action_tr))
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, info in enumerate(infos):
                if info['real_done']:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            storage.store_next(states=toTensor(states),
                               actions=action_tr,
                               values=v_tr,
                               log_pi=log_prob_tr,
                               entropy=entropy_tr,
                               rewards=toTensor(rewards).unsqueeze(-1),
                               terminals=toTensor(1 - terminals).unsqueeze(-1))
            states = config.state_normalizer(next_states)
        
        self.states = states
        action_tr, log_prob_tr, entropy_tr, v_tr = self.network(states)
        storage.values.append(v_tr)
        
        advantages = toTensor(np.zeros((config.num_workers, 1)))
        returns = v_tr.detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.rewards[i] + config.discount * storage.terminals[i] * returns
            if not config.use_gae:
                advantages = returns - storage.values[i]
            else:
                td_error = storage.rewards[i] + config.discount * storage.terminals[i] * storage.values[i + 1] \
                           - storage.values[i]
                advantages = storage.terminals[i] * config.gae_tau * config.discount * advantages + td_error
            storage.advantages[i] = advantages.detach()
            storage.returns[i] = returns.detach()
        
        states, actions, log_prob_old, returns, advantages = storage.cat(
            ['states', 'actions', 'log_pi', 'returns', 'advantages'])
        actions = actions.detach()
        log_prob_old = log_prob_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()
        
        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = toTensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_prob_old = log_prob_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                
                action_tr, log_prob_tr, entropy_tr, v_tr = self.network(sampled_states, sampled_actions)
                ratio = (log_prob_tr - sampled_log_prob_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - config.ppo_ratio_clip,
                                          1.0 + config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() \
                              - config.entropy_weight * entropy_tr.mean()
                value_loss = 0.5 * (sampled_returns - v_tr).pow(2).mean()
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.optimizer.step()
        
        self.total_steps += config.rollout_length * config.num_workers
