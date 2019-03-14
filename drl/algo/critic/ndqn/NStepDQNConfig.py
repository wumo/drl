from drl.common.Config import Config

class NStepDQNConfig(Config):
    def __init__(self):
        super().__init__()
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.history_length = None
        self.double_q = False
        self.tag = 'dqn'
        self.random_action_prob = None
