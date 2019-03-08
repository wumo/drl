from drl.common.Normalizer import RescaleNormalizer

class DQNConfig:
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.network_fn = None
        self.replay_fn = None
        self.discount = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.logger = None
        self.history_length = None
        self.double_q = False
        self.tag = 'dqn'
        self.num_workers = 1
        self.gradient_clip = None
        self.target_network_mix = 0.001
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.max_steps = 0
        self.rollout_length = None
        self.iteration_log_interval = 30
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.__eval_env = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.async_actor = True
    
    @property
    def eval_env(self):
        return self.__eval_env
    
    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name
