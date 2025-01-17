from drl.algo.BaseAgent import BaseAgent
from drl.common.StorageBuffer import StorageBuffer
from drl.util.torch_utils import toNumpy, range_tensor, toTensor
import numpy as np
from torch.nn.utils import clip_grad_norm_

class VPGStorageBuffer(StorageBuffer):
    def __init__(self, size):
        StorageBuffer.__int__(self, size)
        self.states = [None] * size
        self.actions = [None] * size
        self.rewards = [None] * size
        self.values = [None] * size
        self.log_pi = [None] * size
        self.entropy = [None] * size
        self.terminals = [None] * size
        self.advantages = [None] * size
        self.returns = [None] * size

class VPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        
        self.network = config.network_fn()
        self.actor_optimizer = config.optimizer_fn(self.network.actor_params + self.network.shared_params)
        self.critic_optimizer = config.optimizer_fn(self.network.critic_params + self.network.shared_params)
        
        self.task = config.task_fn()
        self.states = self.task.reset()
        
        self.online_rewards = np.zeros(config.num_workers)
    
    def step(self):
        config = self.config
        storage = VPGStorageBuffer(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            action_tr, log_prob_tr, entropy_tr, v_tr = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, _ = self.task.step(toNumpy(action_tr))
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            storage.store_next(states=toTensor(states),
                               actions=action_tr,
                               values=v_tr,
                               log_pi=log_prob_tr,
                               entropy=entropy_tr,
                               rewards=toTensor(rewards).unsqueeze(-1),
                               terminals=toTensor(1 - terminals).unsqueeze(-1))
            states = next_states
        
        self.states = states
        action_tr, log_prob_tr, entropy_tr, v_tr = self.network(config.state_normalizer(states))
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
        
        policy_loss = -(log_prob_old * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss - config.entropy_weight * entropy_loss + config.value_loss_weight * value_loss
        self.actor_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.actor_optimizer.step()
        
        self.total_steps += config.rollout_length * config.num_workers
