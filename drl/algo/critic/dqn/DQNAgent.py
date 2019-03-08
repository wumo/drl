from drl.algo.BaseAgent import BaseAgent, BaseActor
from drl.util.torch_utils import to_np, range_tensor
import numpy as np

class DQNActor(BaseActor):
    def __init__(self, config):
        super().__init__(self, config)
    
    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        q_values = self._network(config.state_normalizer(self._state))
        q_values = to_np(q_values).flatten()
        action = np.random.randint(0, len(q_values)) \
            if self._total_steps < config.exploration_steps or np.random.rand() < config.random_action_prob() \
            else np.max(q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry

class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(self, config)
        self.config = config
        
        self.reply = config.replay_fn()
        self.actor = DQNActor(config)
        
        self.network = config.config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        
        self.actor.set_network(self.network)
        
        self.episode_reward = 0
        self.episode_rewards = []
        
        self.total_steps = 0
        self.batch_indices = range_tensor(self.reply.batch_size)
    
    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action
    
    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, _ in transitions:
            self.episode_reward += reward
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            if done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
            experiences.append([state, action, reward, next_state, done])
        self.reply.store_batch(experiences)
        
        if self.total_steps > self.config.exploration_steps:
            experiences=self.reply.sample()
