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
        return [prediction, self._states, rewards, next_states, terminals]

class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        
        self.actor = A2CActor(config)
        self.actor.set_network(self.network)
        self.buffer = AdvantageBuffer(config)
        
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
        transitions = self.actor.step()
        for prediction, states, rewards, next_states, terminals, _ in transitions:
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            self.buffer.store(states, toNumpy(prediction['a']), rewards, toNumpy(prediction['v']),
                              toNumpy(prediction['log_pi_a']))
        
        states, actions, advantages, returns, log_pis = self.buffer.get()
        
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
