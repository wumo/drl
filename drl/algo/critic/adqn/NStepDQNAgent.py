from drl.algo.BaseAgent import BaseAgent
from drl.util.torch_utils import toNumpy, range_tensor, toTensor, epsilon_greedy, huber_loss
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# Asynchronous Methods for Deep Reinforcement Learning
class NStepDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        
        self.task = config.task_fn()
        self.states = self.task.reset()
        self.online_rewards = np.zeros(config.num_workers)
    
    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(toNumpy(q))
        self.config.state_normalizer.unset_read_only()
        return action
    
    def step(self):
        config = self.config
        
        rollout = []
        states = self.states
        for _ in range(self.config.rollout_length):
            # choose according to max(Q)
            q = self.network(config.state_normalizer(states))
            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, toNumpy(q))
            
            next_states, rewards, terminals, _ = self.task.step(actions)
            self.online_rewards += rewards
            
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            
            rollout.append([q, actions, rewards, 1 - terminals])
            states = next_states
            
            self.total_steps += config.num_workers
            if self.total_steps / config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
        
        self.states = states
        
        processed_rollout = [None] * len(rollout)
        returns = self.target_network(config.state_normalizer(states)).detach()
        returns, _ = torch.max(returns, dim=-1, keepdim=True)
        for i in reversed(range(len(rollout))):
            q, actions, rewards, terminals = rollout[i]
            actions = toTensor(actions).unsqueeze(1).long()
            q = q.gather(1, actions)
            terminals = toTensor(terminals).unsqueeze(1)
            rewards = toTensor(rewards).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns
            processed_rollout[i] = [q, returns]
        
        q, returns = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        # loss = F.smooth_l1_loss(q, returns)
        # loss = huber_loss(q - returns)
        loss = 0.5 * (q - returns).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
        
        if self.total_steps / config.rollout_length % config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
