import numpy as np
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from ..util.torch_utils import random_seed
from .SingleProcessVecEnv import SingleProcessVecEnv
from .SubProcessesVecEnv import SubProcessesVecEnv
# from drl.bench.monitor import Monitor
from drl.util.misc import mkdir
from drl.environment.atari_wrappers import make_atari, wrap_deepmind
from drl.environment.atari_wrappers import FrameStack

class Monitor(gym.core.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        info['real_done'] = done
        return ob, rew, done, info

def configure_env_maker(env_name, rank, log_dir=None, seed=np.random.randint(int(1e5)),
                        episode_life=True, history_length=4):
    def maker():
        random_seed(seed)
        env = gym.make(env_name)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_name)
        env.seed(seed + rank)
        env = Monitor(env)
        if is_atari:
            env = wrap_deepmind(env, episode_life=episode_life, frame_stack=True, history_length=history_length)
        return env
    
    return maker

class Task:
    def __init__(self, env_name, num_envs=1, single_process=True, log_dir=None,
                 **kwargs):
        if log_dir is not None:
            mkdir(log_dir)
        env_makers = [configure_env_maker(env_name, i, log_dir, **kwargs) for i in range(num_envs)]
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
