from drl.algo.actor_critic.ddpg.DDPGAgent import DDPGAgent
from drl.algo.actor_critic.ddpg.DDPGConfig import DDPGConfig
from drl.environment.Task import Task
from drl.common.ReplayBuffer import ReplayBuffer
from drl.common.RandomProcess import OrnsteinUhlenbeckProcess
from drl.common.Schedule import LinearSchedule
from torch.optim import Adam, RMSprop
from drl.network.network_heads import DeterministicActorCriticNet
from drl.network.network_bodies import FCBody, TwoLayerFCBodyWithAction
import torch.nn.functional as F
from drl.util.torch_utils import random_seed, select_device

def ddpg_continuous(game, tag=""):
    config = DDPGConfig()
    config.task_fn = lambda: Task(game)
    config.eval_env = Task(game)
    
    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: Adam(params, lr=1e-3))
    
    config.batch_size = 64
    config.replay_fn = lambda: ReplayBuffer(config.eval_env, memory_size=int(1e6))
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    
    config.discount = 0.99
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.max_steps = int(1e6)
    DDPGAgent(config).run_steps(tag=f'{tag}{ddpg_continuous.__name__}-{game}')

if __name__ == '__main__':
    random_seed()
    select_device(0)
    game = 'HalfCheetah-v2'
    ddpg_continuous(game, "bench-")
