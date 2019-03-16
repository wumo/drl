from drl.common.Normalizer import RescaleNormalizer

class AgentConfig:
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.network_fn = None
        self.discount = None
        self.logger = None
        self.tag = 'a2c'
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
        self.__eval_env = None
        self.log_interval = int(2e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        
        self.gc_interval = 0
    
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
