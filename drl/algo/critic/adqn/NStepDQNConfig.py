from drl.common.Normalizer import RescaleNormalizer

class NStepDQNConfig:
    def __init__(self):
        self.log_dir = None
        self.task_fn = None
        self.optimizer_fn = None
        self.network_fn = None
        self.discount = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.logger = None
        self.history_length = None
        self.double_q = False
        self.tag = 'dqn'
        self.state_dim = None
        self.state_shape = None
        self.action_dim = None
        self.action_shape = None
        self.task_name = None
        self.num_workers = 1
        self.gradient_clip = None
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.max_steps = 0
        self.rollout_length = None
        self.random_action_prob = None
        self.__eval_env = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
    
    @property
    def eval_env(self):
        return self.__eval_env
    
    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.state_shape = env.state_shape
        self.action_dim = env.action_dim
        self.action_shape = env.action_shape
        self.task_name = env.name
