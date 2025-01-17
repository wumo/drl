from drl.common.Normalizer import RescaleNormalizer

class VPGConfig:
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.network_fn = None
        self.discount = None
        self.logger = None
        self.history_length = None
        self.tag = 'a2c'
        self.state_dim = None
        self.action_dim = None
        self.task_name = None
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.value_loss_weight = 1.0
        self.use_gae = False
        self.gae_tau = 1.0
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.max_steps = 0
        self.rollout_length = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
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
