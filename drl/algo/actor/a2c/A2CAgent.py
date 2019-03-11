from drl.algo.BaseAgent import BaseAgent, BaseActor
from drl.common.StorageBuffer import StorageBuffer
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
        action, log_prob, entropy, v = self._network(config.state_normalizer(self._states))
        next_states, rewards, terminals, _ = self._task.step(toNumpy(action))
        
        self._total_steps += 1
        self._states = next_states
        return [action, log_prob, entropy, v, self._states, rewards, next_states, terminals]

class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        
        self.actor = A2CActor(config)
        self.actor.set_network(self.network)
        
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
        storage = StorageBuffer({'states', 'actions', 'rewards', 'values', 'log_pi', 'entropy', 'terminals'})
        transitions = self.actor.step()
        for action, log_prob, entropy, v, states, rewards, next_states, terminals in transitions:
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            storage.append(states=states,
                           actions=toNumpy(action),
                           rewards=rewards,
                           values=toNumpy(v),
                           log_pi=toNumpy(log_prob),
                           entropy=toNumpy(entropy),
                           terminals=terminals)
        