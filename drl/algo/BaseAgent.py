import torch
import numpy as np

class BaseActor:
    def __init__(self, config):
        self.config = config
        
        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        
        self.step = self._sample
        self._set_up()
        self._task = config.task_fn()
    
    def _sample(self):
        transitions = []
        for _ in range(self.config.rollout_length):
            transitions.append(self._transition())
        return transitions
    
    def _transition(self):
        raise NotImplementedError
    
    def _set_up(self):
        pass
    
    def set_network(self, net):
        self._network = net

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.total_steps = 0
        self.episode_rewards = []
    
    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
    
    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
    
    def eval_step(self, state):
        raise NotImplementedError
    
    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        total_rewards = 0
        while True:
            action = self.eval_step(state)
            state, reward, done, _ = env.step([action])
            total_rewards += reward[0]
            if done[0]:
                break
        return total_rewards
    
    def eval_episodes(self):
        rewards = []
        for ep in range(self.config.eval_episodes):
            rewards.append(self.eval_episode())
        self.config.logger.info(f'evaluation episode return: {np.mean(rewards)}'
                                f'({np.std(rewards) / np.sqrt(len(rewards))})')
    
    def step(self):
        raise NotImplementedError
