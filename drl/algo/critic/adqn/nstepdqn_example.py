from drl.algo.critic.adqn.NStepDQNConfig import NStepDQNConfig
from drl.algo.critic.adqn.NStepDQNAgent import NStepDQNAgent
from drl.environment.Task import Task
from torch.optim import RMSprop
from drl.network.network_heads import VanillaNet
from drl.network.network_bodies import FCBody
from drl.common.ReplayBuffer import ReplayBuffer
from drl.common.Schedule import LinearSchedule
from drl.common.run_utils import run_steps
from drl.util.logger import get_logger
from drl.util.torch_utils import random_seed, select_device
from drl.common.DeviceSetting import DEVICE
import torch

def nstepdqn_cart_pole(game):
    config = NStepDQNConfig()
    config.num_workers = 24
    config.task_fn = lambda: Task(game, num_envs=config.num_workers, single_process=False)
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
    config.logger = get_logger(nstepdqn_cart_pole.__name__)
    run_steps(NStepDQNAgent(config))

def nstepdqn_cart_pole_image(game):
    config = NStepDQNConfig()
    config.num_workers = 24
    config.task_fn = lambda: Task(game, num_envs=config.num_workers)
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
    config.logger = get_logger(nstepdqn_cart_pole.__name__)
    run_steps(NStepDQNAgent(config))

def nstepdqn_pixel_atari(game):
    config = NStepDQNConfig()
    config.task_fn = lambda: Task(game)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    
    config.replay_fn = lambda: ReplayBuffer(config.state_dim, 0,
                                            memory_size=int(1e6), batch_size=32)
    
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    # config.double_q = True
    config.double_q = False
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    config.logger = get_logger(nstepdqn_cart_pole.__name__)
    run_steps(NStepDQNAgent(config))

if __name__ == '__main__':
    random_seed()
    select_device(-1)
    # game = 'MountainCar-v0'
    game = 'CartPole-v0'
    # game = 'BreakoutNoFrameskip-v4'
    nstepdqn_cart_pole(game)
