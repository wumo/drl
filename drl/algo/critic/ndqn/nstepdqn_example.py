from drl.algo.critic.ndqn.NStepDQNConfig import NStepDQNConfig
from drl.algo.critic.ndqn.NStepDQNAgent import NStepDQNAgent
from drl.environment.Task import Task
from torch.optim import RMSprop
from drl.network.network_heads import VanillaNet, NatureConvBody, DQNBody, DuelingNet
from drl.network.network_bodies import FCBody
from drl.common.Normalizer import ImageNormalizer, SignNormalizer
from drl.common.Schedule import LinearSchedule
from drl.util.logger import get_logger
from drl.util.torch_utils import random_seed, select_device

def nstepdqn_cart_pole():
    game = 'CartPole-v0'
    config = NStepDQNConfig()
    config.num_workers = 16
    config.task_fn = lambda: Task(game, num_envs=config.num_workers, single_process=True)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.double_q = True
    config.rollout_length = 1
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = 1e6
    config.logger = get_logger(tag=f'{nstepdqn_pixel_atari.__name__}-{game}')
    NStepDQNAgent(config).run_steps()

def nstepdqn_pixel_atari(game):
    config = NStepDQNConfig()
    config.num_workers = 16
    config.task_fn = lambda: Task(game, num_envs=config.num_workers, single_process=False)
    config.eval_env = Task(game)
    
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    
    config.optimizer_fn = lambda params: RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody())
    # config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody())
    
    config.random_action_prob = LinearSchedule(1.0, 0.05, 1e6)
    config.discount = 0.99
    
    config.target_network_update_freq = 10000
    config.double_q = True
    # config.double_q = False
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    
    # config.eval_interval = int(1e4)
    # config.eval_episodes = 10
    config.logger = get_logger(tag=f'{nstepdqn_pixel_atari.__name__}-{game}')
    NStepDQNAgent(config).run_steps()

if __name__ == '__main__':
    random_seed()
    select_device(0)
    # game = 'MountainCar-v0'
    # game = 'CartPole-v0'
    game = 'BreakoutNoFrameskip-v4'
    # nstepdqn_cart_pole()
    nstepdqn_pixel_atari(game)
