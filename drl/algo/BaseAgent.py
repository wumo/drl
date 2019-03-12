import torch
import numpy as np

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
        pass
    
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
        mean_rewards,std=np.mean(rewards),np.std(rewards) / np.sqrt(len(rewards))
        self.config.logger.info(f'evaluation episode return: {mean_rewards}'
                                f'({std})')
        self.config.logger.add_scalar("eval_mean_rewards", mean_rewards, self.total_steps)
    
    def step(self):
        raise NotImplementedError

    def close(self):
        self.task.close()