from drl.algo.AgentConfig import AgentConfig

class QuantileRegressionDQNConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        self.tag = 'qrdqn'
        self.replay_fn = None
        self.random_action_prob = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.history_length = None
        self.batch_size = None
        self.num_quantiles = None
        
        
