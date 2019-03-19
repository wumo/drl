from drl.algo.AgentConfig import AgentConfig

class CategoricalDQNConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        self.tag = 'dqn'
        self.replay_fn = None
        self.random_action_prob = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.history_length = None
        self.batch_size = None
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        
        
