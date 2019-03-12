import numpy as np
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from ..util.torch_utils import random_seed
from .SingleProcessVecEnv import SingleProcessVecEnv
from .SubProcessesVecEnv import SubProcessesVecEnv
from drl.bench.monitor import Monitor
from os.path import join
from drl.util.misc import mkdir

def configure_env_maker(env_name, seed, rank, log_dir=None):
    def maker():
        random_seed(seed)
        env = gym.make(env_name)
        env.seed(seed + rank)
        if log_dir is not None:
            env = Monitor(env, filename=join(log_dir, str(rank)), allow_early_resets=True)
        return env
    
    return maker

class Task:
    def __init__(self, env_name, num_envs=1, single_process=True, log_dir=None, seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)
        env_makers = [configure_env_maker(env_name, seed, i, log_dir) for i in range(num_envs)]
        Wrapper = SingleProcessVecEnv if single_process else SubProcessesVecEnv
        self.env = Wrapper(env_makers)
        self.name = env_name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))
        self.state_shape = self.observation_space.shape
        
        self.action_space = self.env.action_space
        self.action_shape = self.action_space.shape
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'
    
    def reset(self):
        return self.env.reset()
    
    def close(self):
        self.env.close()
    
    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)
