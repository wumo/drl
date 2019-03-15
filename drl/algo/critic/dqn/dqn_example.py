from drl.algo.critic.dqn.DQNConfig import DQNConfig
from drl.algo.critic.dqn.DQNAgent import DQNAgent
from drl.environment.Task import Task
from torch.optim import RMSprop
from drl.network.network_heads import VanillaNet, DuelingNet, NatureConvBody
from drl.network.network_bodies import FCBody
from drl.common.ReplayBuffer import ReplayBuffer
from drl.common.Schedule import LinearSchedule
from drl.util.logger import get_logger
from drl.util.torch_utils import random_seed, select_device

def dqn_cart_pole(game):
    config = DQNConfig()
    config.task_fn = lambda: Task(game)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    
    config.replay_fn = lambda: ReplayBuffer(config.state_shape, config.action_shape,
                                            memory_size=int(1e4), batch_size=10)
    
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    config.double_q = True
    config.rollout_length = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = 1e6
    config.logger = get_logger(dqn_cart_pole.__name__)
    DQNAgent(config).run_steps()

def dqn_pixel_atari(game):
    config = DQNConfig()
    config.history_length = 4
    config.task_fn = lambda: Task(game)
    config.eval_env = Task(game, episode_life=False)
    
    config.optimizer_fn = lambda params: RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    # config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    
    config.replay_fn = lambda: ReplayBuffer(config.state_shape, config.action_shape,
                                            memory_size=int(1e6), batch_size=32)
    
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 10000
    config.double_q = True
    # config.double_q = False
    config.rollout_length = 4
    config.gradient_clip = 5
    config.max_steps = 2e7
    config.logger = get_logger(dqn_cart_pole.__name__)
    DQNAgent(config).run_steps()

if __name__ == '__main__':
    random_seed()
    select_device(0)
    # game = 'MountainCar-v0'
    game = 'CartPole-v0'
    # game = 'BreakoutNoFrameskip-v4'
    dqn_cart_pole(game)
