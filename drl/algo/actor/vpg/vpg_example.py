from drl.algo.actor.vpg.VPGAgent import VPGAgent
from drl.algo.actor.vpg.VPGConfig import VPGConfig
from drl.environment.Task import Task
from torch.optim import Adam
from drl.network.network_heads import CategoricalActorCriticNet
from drl.network.network_bodies import FCBody
from drl.common.run_utils import run_steps
from drl.util.logger import get_logger
from drl.util.torch_utils import random_seed, select_device

def vpg_cart_pole(game):
    config = VPGConfig()
    config.num_workers = 5
    config.task_fn = lambda: Task(game, num_envs=config.num_workers)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: Adam(params, lr=1e-3)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim))
    
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.97
    config.entropy_weight = 0.001
    config.rollout_length = 4000
    config.gradient_clip = 5
    config.logger = get_logger(tag=vpg_cart_pole.__name__)
    run_steps(VPGAgent(config))

if __name__ == '__main__':
    random_seed(0)
    select_device(0)
    # game = 'MountainCar-v0'
    game = 'CartPole-v0'
    # game = 'BreakoutNoFrameskip-v4'
    vpg_cart_pole(game)
