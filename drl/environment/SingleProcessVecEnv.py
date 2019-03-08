from .VecEnv import VecEnv
import numpy as np

class SingleProcessVecEnv(VecEnv):
    def __init__(self, env_makers):
        self.envs = [env_maker() for env_maker in env_makers]
        first_env = self.envs[0]
        VecEnv.__init__(self, len(self.envs), first_env.observation_space, first_env.action_space)
        self.actions = None
    
    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step_async(self, actions):
        self.actions = actions
    
    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info