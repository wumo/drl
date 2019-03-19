from drl.algo.critic.dqn.DQNConfig import DQNConfig
from drl.algo.critic.dqn.DQNAgent import DQNAgent
from drl.environment.Task import Task
from torch.optim import RMSprop
from drl.network.network_heads import VanillaNet, DuelingNet, NatureConvBody
from drl.network.network_bodies import FCBody
from drl.common.ReplayBuffer import ReplayBuffer
from drl.common.Schedule import LinearSchedule
from drl.common.Normalizer import ImageNormalizer, SignNormalizer
from drl.util.torch_utils import random_seed, select_device

def dqn_cart_pole():
    game = 'CartPole-v0'
    config = DQNConfig()
    config.task_fn = lambda: Task(game)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    
    config.batch_size = 32
    config.replay_fn = lambda: ReplayBuffer(config.eval_env, memory_size=int(1e4))
    
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    config.double_q = True
    config.rollout_length = 4
    config.gradient_clip = 5
    config.max_steps = 1e6
    DQNAgent(config).run_steps(tag=f'{dqn_cart_pole.__name__}-{game}')

def dqn_pixel_atari(game, tag=""):
    config = DQNConfig()
    config.history_length = 4
    
    config.task_fn = lambda: Task(game)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    # config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody())
    
    config.batch_size = 32
    config.replay_fn = lambda: ReplayBuffer(config.eval_env, memory_size=int(1e6), stack=config.history_length)
    
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 10000
    config.double_q = True
    # config.double_q = False
    config.rollout_length = 4
    config.gradient_clip = 5
    config.max_steps = 2e7
    DQNAgent(config).run_steps(tag=f'{tag}{dqn_pixel_atari.__name__}-{game}')

if __name__ == '__main__':
    random_seed()
    select_device(0)
    # game = 'MountainCar-v0'
    
    game = 'BreakoutNoFrameskip-v4'
    dqn_cart_pole()
    # dqn_pixel_atari(game, "bench-")
