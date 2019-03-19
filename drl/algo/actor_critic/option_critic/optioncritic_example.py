from drl.algo.actor_critic.option_critic.OptionCriticConfig import OptionCriticConfig
from drl.algo.actor_critic.option_critic.OptionCriticAgent import OptionCriticAgent
from drl.environment.Task import Task
from torch.optim import Adam, RMSprop
from drl.network.network_heads import OptionCriticNet, GaussianActorCriticNet
from drl.network.network_bodies import FCBody, NatureConvBody
from drl.common.Normalizer import ImageNormalizer, SignNormalizer
from drl.common.Schedule import LinearSchedule
from drl.util.logger import get_logger
from drl.util.torch_utils import random_seed, select_device
import torch.nn.functional as F

def option_critic_cart_pole():
    game = 'CartPole-v0'
    config = OptionCriticConfig()
    config.num_workers = 8
    config.task_fn = lambda: Task(game, num_envs=config.num_workers, single_process=True)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: RMSprop(params, 0.001)
    config.network_fn = lambda: OptionCriticNet(FCBody(config.state_dim), config.action_dim, num_options=2)
    config.random_option_prob = LinearSchedule(1.0, 0.01, 1e4)
    
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.max_steps = 1e6
    OptionCriticAgent(config).run_steps(tag=f'{option_critic_cart_pole.__name__}-{game}')

def option_critic_pixel_atari(game, tag=""):
    config = OptionCriticConfig()
    config.history_length = 4
    config.num_workers = 16
    config.task_fn = lambda: Task(game, num_envs=config.num_workers, single_process=False,
                                  history_length=config.history_length)
    config.eval_env = Task(game, episode_life=False, history_length=config.history_length)
    
    config.optimizer_fn = lambda params: RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: OptionCriticNet(NatureConvBody(in_channels=config.history_length),
                                                config.action_dim, num_options=4)
    config.random_option_prob = LinearSchedule(0.1)
    
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    OptionCriticAgent(config).run_steps(tag=f'{tag}{option_critic_pixel_atari.__name__}-{game}')

if __name__ == '__main__':
    random_seed()
    select_device(0)
    # game = 'MountainCar-v0'
    game = 'BreakoutNoFrameskip-v4'
    # option_critic_cart_pole()
    option_critic_pixel_atari(game,"bench-")
