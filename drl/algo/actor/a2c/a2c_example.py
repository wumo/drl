from drl.algo.actor.a2c.A2CAgent import A2CAgent
from drl.algo.actor.a2c.A2CConfig import A2CConfig
from drl.environment.Task import Task
from torch.optim import Adam
from drl.network.network_heads import CategoricalActorCriticNet
from drl.network.network_bodies import FCBody
from drl.common.run_utils import run_steps
from drl.util.logger import get_logger
from drl.util.torch_utils import random_seed, select_device

def a2c_cart_pole(game):
    config = A2CConfig()
    config.num_workers = 5
    config.task_fn = lambda: Task(game, num_envs=config.num_workers)
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
    config.logger = get_logger(tag=a2c_cart_pole.__name__)
    run_steps(A2CAgent(config))

if __name__ == '__main__':
    random_seed(0)
    select_device(0)
    # game = 'MountainCar-v0'
    game = 'CartPole-v0'
    # game = 'BreakoutNoFrameskip-v4'
    a2c_cart_pole(game)
