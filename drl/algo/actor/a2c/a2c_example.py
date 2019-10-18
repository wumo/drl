from drl.algo.actor.a2c.A2CAgent import A2CAgent
from drl.algo.actor.a2c.A2CConfig import A2CConfig
from drl.environment.Task import Task
from torch.optim import Adam, RMSprop
from drl.network.network_heads import CategoricalActorCriticNet, GaussianActorCriticNet
from drl.network.network_bodies import FCBody, NatureConvBody
from drl.common.Normalizer import ImageNormalizer, SignNormalizer
from drl.util.logger import get_logger
from drl.util.torch_utils import random_seed, select_device

def a2c_cart_pole():
    game = 'CartPole-v0'
    config = A2CConfig()
    config.num_workers = 16
    config.task_fn = lambda: Task(game, num_envs=config.num_workers, single_process=True)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: Adam(params, lr=1e-3)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim))
    
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.max_steps = 1e6
    A2CAgent(config).run_steps(tag=f'{a2c_cart_pole.__name__}-{game}')

def a2c_pixel_atari(game, tag=""):
    config = A2CConfig()
    config.num_workers = 16
    config.task_fn = lambda: Task(game, num_envs=config.num_workers, single_process=False)
    config.eval_env = Task(game, episode_life=False)
    
    config.optimizer_fn = lambda params: RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
    
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    A2CAgent(config).run_steps(tag=f'{tag}{a2c_pixel_atari.__name__}-{game}')

def a2c_continuous(game, tag=""):
    config = A2CConfig()
    config.num_workers = 16
    config.task_fn = lambda: Task(game, num_envs=config.num_workers, single_process=True)
    config.eval_env = Task(game)
    config.optimizer_fn = lambda params: RMSprop(params, lr=0.0007)
    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim), critic_body=FCBody(config.state_dim))
    
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(1e6)
    A2CAgent(config).run_steps(tag=f'{tag}{a2c_continuous.__name__}-{game}')

if __name__ == '__main__':
    random_seed()
    select_device(0)
    # game = 'MountainCar-v0'
    # game = 'CartPole-v0'
    game = 'BreakoutNoFrameskip-v4'
    a2c_cart_pole()
    # game = 'Reacher-v2'
    # a2c_pixel_atari(game, "bench")
    # a2c_continuous(game)
    a2c_cart_pole()
