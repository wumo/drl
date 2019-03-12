from drl.algo.BaseAgent import BaseAgent
from drl.util.torch_utils import toNumpy, range_tensor, toTensor
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

# Human-level control through deep reinforcement (DQN)
class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        
        self.reply = config.replay_fn()
        
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        
        self.task = config.task_fn()
        self.states = self.task.reset()
        self.online_reward = 0
        
        self.batch_indices = range_tensor(self.reply.batch_size)
    
    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(toNumpy(q))
        self.config.state_normalizer.unset_read_only()
        return action
    
    def step(self):
        config = self.config
        
        # rollout
        for _ in range(self.config.rollout_length):
            # choose according to max(Q)
            q = self.network(config.state_normalizer(self.states))
            q = toNumpy(q)
            action = np.random.randint(0, len(q)) \
                if self.total_steps < config.exploration_steps or np.random.rand() < config.random_action_prob() \
                else np.argmax(q)
            
            next_states, rewards, dones, info = self.task.step([action])
            state, reward, next_state, done = self.states[0], rewards[0], next_states[0], int(dones[0])
            self.states = next_states
            self.total_steps += 1
            self.online_reward += reward
            
            reward = config.reward_normalizer(reward)
            if done:
                self.episode_rewards.append(self.online_reward)
                self.online_reward = 0
            self.reply.store([state, action, reward, next_state, done])
        
        if self.total_steps > config.exploration_steps:
            # minibatch gradient descent
            experiences = self.reply.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = config.state_normalizer(states)
            next_states = config.state_normalizer(next_states)
            q_next = self.target_network(next_states).detach()
            if config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            terminals = toTensor(terminals)
            rewards = toTensor(rewards)
            q_next = rewards + config.discount * q_next * (1 - terminals)
            
            actions = toTensor(actions).long()
            q = self.network(states)
            q = q[self.batch_indices, actions]
            
            loss = (q_next - q).pow(2).mul(0.5).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.network.parameters(), config.gradient_clip)
            self.optimizer.step()
        
        if self.total_steps / config.rollout_length % config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
