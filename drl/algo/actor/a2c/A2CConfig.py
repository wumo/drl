from drl.algo.AgentConfig import AgentConfig

class A2CConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        self.history_length = 4
        self.tag = 'a2c'
        self.value_loss_weight = 1.0
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
