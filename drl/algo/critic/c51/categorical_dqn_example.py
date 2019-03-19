from drl.algo.critic.c51.CategoricalDQNConfig import CategoricalDQNConfig
from drl.algo.critic.c51.CategoricalDQNAgent import CategoricalDQNAgent
from drl.environment.Task import Task
from torch.optim import RMSprop, Adam
from drl.network.network_heads import CategoricalNet, NatureConvBody
from drl.network.network_bodies import FCBody
from drl.common.ReplayBuffer import ReplayBuffer
from drl.common.Schedule import LinearSchedule
from drl.common.Normalizer import ImageNormalizer, SignNormalizer
from drl.util.torch_utils import random_seed, select_device

def categorical_dqn_cart_pole():
    game = 'CartPole-v0'
    config = CategoricalDQNConfig()
    config.task_fn = lambda: Task(game)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms,
                                               FCBody(config.state_dim))
    
    config.batch_size = 10
    config.replay_fn = lambda: ReplayBuffer(config.eval_env, memory_size=int(1e4))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.categorical_v_max = 100
    config.categorical_v_min = -100
    config.categorical_n_atoms = 50
    config.rollout_length = 4
    config.gradient_clip = 5
    
    config.max_steps = 1e5
    CategoricalDQNAgent(config).run_steps(tag=f'{categorical_dqn_cart_pole.__name__}-{game}')

def categorical_dqn_pixel_atari(game, tag=""):
    config = CategoricalDQNConfig()
    config.history_length = 4
    
    config.task_fn = lambda: Task(game)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms,
                                               NatureConvBody())
    
    config.batch_size = 32
    config.replay_fn = lambda: ReplayBuffer(config.eval_env, memory_size=int(1e6), stack=config.history_length)
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)
    
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    config.rollout_length = 4
    config.gradient_clip = 0.5
    config.max_steps = 2e7
    CategoricalDQNAgent(config).run_steps(tag=f'{tag}{categorical_dqn_pixel_atari.__name__}-{game}')

if __name__ == '__main__':
    random_seed()
    select_device(0)
    # game = 'MountainCar-v0'
    
    game = 'BreakoutNoFrameskip-v4'
    # categorical_dqn_cart_pole()
    categorical_dqn_pixel_atari(game, "bench-")
