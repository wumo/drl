from drl.algo.BaseAgent import BaseAgent, BaseActor
from drl.algo.actor.a2c.AdvantageBuffer import AdvantageBuffer
from drl.util.torch_utils import toNumpy, range_tensor, toTensor
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

class A2CActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
    
    def _transition(self):
        if self._states is None:
            self._states = self._task.reset()
        config = self.config
        prediction = self._network(config.state_normalizer(self._states))
        next_states, rewards, terminals, _ = self._task.step(toNumpy(prediction['a']))
        
        self._total_steps += 1
        self._states = next_states
        return entry

class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor = A2CActor(config)
        self.actor.set_network(self.network)
        self.buffer = AdvantageBuffer(config)
        
        self.episode_rewards = []
        self.online_reward = np.zeros(config.num_workers)
        
        self.total_steps = 0
        self.batch_indices = range_tensor(self.reply.batch_size)
    
    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(toNumpy(q))
        self.config.state_normalizer.unset_read_only()
        return action
    
    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, _ in transitions:
            self.episode_reward += reward
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            if done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
            experiences.append([state, action, reward, next_state, done])
        self.reply.store_batch(experiences)
        
        if self.total_steps > config.exploration_steps:
            # minibatch gradient descent
            experiences = self.reply.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = config.state_normalizer(states)
            next_states = config.state_normalizer(next_states)
            q_next = self.target_network(next_states).detach()
            if config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            terminals = toTensor(terminals)
            rewards = toTensor(rewards)
            q_next = rewards + config.discount * q_next * (1 - terminals)
            
            actions = toTensor(actions).long()
            q = self.network(states)
            q = q[self.batch_indices, actions]
            
            loss = (q_next - q).pow(2).mul(0.5).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.network.parameters(), config.gradient_clip)
            self.optimizer.step()
        
        if self.total_steps / config.sgd_update_frequency % config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
