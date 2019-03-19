from drl.algo.AgentConfig import AgentConfig

class DDPGConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        self.tag = 'ddpg'
        self.random_process_fn = None
        self.min_memory_size = None
        self.target_network_mix = 0.001
        
