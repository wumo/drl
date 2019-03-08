from drl.algo.critic.dqn.DQNConfig import DQNConfig
from drl.algo.critic.dqn.DQNAgent import DQNAgent
from drl.environment.Task import Task
from torch.optim import RMSprop
from drl.network.network_heads import VanillaNet
from drl.network.network_bodies import FCBody
from drl.common.ReplayBuffer import ReplayBuffer
from drl.common.Schedule import LinearSchedule
from drl.common.run_utils import run_steps
from drl.util.logger import get_logger

def dqn_cart_pole():
    game = 'CartPole-v0'
    config = DQNConfig()
    config.task_fn = lambda: Task(game)
    config.eval_env = Task(game)
    
    config.optimizer_fn = lambda params: RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    
    config.replay_fn = lambda: ReplayBuffer(config.state_dim, config.action_dim,
                                            memory_size=int(1e4), batch_size=10)
    
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    # config.async_actor = False
    config.logger = get_logger(tag=dqn_cart_pole.__name__)
    run_steps(DQNAgent(config))

if __name__ == '__main__':
    dqn_cart_pole()
