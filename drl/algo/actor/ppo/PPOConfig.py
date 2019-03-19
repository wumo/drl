from drl.algo.AgentConfig import AgentConfig

class PPOConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        self.history_length = 4
        self.tag = 'ppo'
        self.entropy_weight = 0
        self.use_gae = True
        self.gae_tau = 1.0
        
        self.optimization_epochs = 10
        self.mini_batch_size = 32 * 5
        self.ppo_ratio_clip = 0.2
