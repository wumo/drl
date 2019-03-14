from drl.algo.actor.ppo.PPOAgent import PPOAgent
from drl.algo.actor.ppo.PPOConfig import PPOConfig
from drl.environment.Task import Task
from torch.optim import Adam, RMSprop
from drl.network.network_heads import CategoricalActorCriticNet, GaussianActorCriticNet
from drl.network.network_bodies import FCBody, NatureConvBody
from drl.common.run_utils import run_steps
from drl.common.Normalizer import ImageNormalizer, SignNormalizer
from drl.util.logger import get_logger
from drl.util.torch_utils import random_seed, select_device

def ppo_cart_pole():
    game = 'CartPole-v0'
    config = PPOConfig()
    config.num_workers = 5
    config.task_fn = lambda: Task(game, num_envs=config.num_workers)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim))
    
    config.max_steps = 3e5
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.001
    config.gradient_clip = 5
    config.rollout_length = 128
    config.optimization_epochs = 10
    config.mini_batch_size = 32 * 5
    config.ppo_ratio_clip = 0.2
    config.log_interval = 128 * 5 * 10
    config.logger = get_logger(tag=ppo_cart_pole.__name__)
    run_steps(PPOAgent(config))

def ppo_pixel_atari(game):
    config = PPOConfig()
    config.history_length = 4
    config.num_workers = 16
    config.task_fn = lambda: Task(game, num_envs=config.num_workers, single_process=False,
                                  history_length=config.history_length)
    config.eval_env = Task(game, episode_life=False, history_length=config.history_length)
    
    config.optimizer_fn = lambda params: RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, NatureConvBody(in_channels=config.history_length))
    
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.logger = get_logger(tag=ppo_pixel_atari.__name__)
    run_steps(PPOAgent(config))

def ppo_continuous(game):
    config = PPOConfig()
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
    config.logger = get_logger(tag=ppo_continuous.__name__)
    run_steps(PPOAgent(config))

if __name__ == '__main__':
    random_seed()
    select_device(0)
    # game = 'MountainCar-v0'
    # game = 'BreakoutNoFrameskip-v4'
    game = 'Reacher-v2'
    ppo_cart_pole()
    # a2c_pixel_atari(game)
    # ppo_continuous(game)
