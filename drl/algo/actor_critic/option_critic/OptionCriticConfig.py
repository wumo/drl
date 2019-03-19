from drl.algo.AgentConfig import AgentConfig

class OptionCriticConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        self.history_length = 4
        self.tag = 'option-critic'
        
        self.target_network_update_freq = None
        self.random_option_prob = None
        self.entropy_weight = 0
        self.termination_regularizer = 0
