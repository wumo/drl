from drl.algo.BaseAgent import BaseAgent
from drl.util.torch_utils import toNumpy, range_tensor, toTensor, epsilon_greedy
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

# Human-level control through deep reinforcement (DQN)
class CategoricalDQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        
        self.replay = config.replay_fn()
        
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        
        self.task = config.task_fn()
        self.states = self.task.reset()
        
        self.batch_indices = range_tensor(self.config.batch_size)
        
        self.atoms = toTensor(np.linspace(config.categorical_v_min, config.categorical_v_max,
                                          config.categorical_n_atoms))
        self.delta_atom = (config.categorical_v_max - config.categorical_v_min) / float(config.categorical_n_atoms - 1)
    
    def step(self):
        config = self.config
        
        # rollout
        for _ in range(self.config.rollout_length):
            # choose according to max(Q)
            probs, _ = self.network(config.state_normalizer(self.states))
            q_values = (probs * self.atoms).sum(-1)
            epsilon = config.random_action_prob(config.num_workers)
            actions = epsilon_greedy(epsilon, toNumpy(q_values))
            
            next_states, rewards, dones, infos = self.task.step(actions)
            state, reward, next_state, done, info = self.states[0], rewards[0], next_states[0], int(dones[0]), infos[0]
            self.states = next_states
            self.total_steps += 1
            
            reward = config.reward_normalizer(reward)
            self.replay.store([state, actions[0], reward, next_state, done])
        
        if self.total_steps > config.exploration_steps:
            # minibatch gradient descent
            experiences = self.replay.sample(config.batch_size)
            states, actions, rewards, next_states, terminals = experiences
            states = config.state_normalizer(states)
            next_states = config.state_normalizer(next_states)
            
            prob_next, _ = self.target_network(next_states)
            prob_next = prob_next.detach()
            q_next = (prob_next * self.atoms).sum(-1)
            a_next = torch.argmax(q_next, dim=-1)
            prob_next = prob_next[self.batch_indices, a_next, :]
            
            rewards = toTensor(rewards).unsqueeze(-1)
            terminals = toTensor(terminals).unsqueeze(-1)
            atoms_next = rewards + self.config.discount * (1 - terminals) * self.atoms.view(1, -1)
            
            atoms_next.clamp_(self.config.categorical_v_min, self.config.categorical_v_max)
            b = (atoms_next - self.config.categorical_v_min) / self.delta_atom
            l = b.floor()
            u = b.ceil()
            d_m_l = (u + (l == u).float() - b) * prob_next
            d_m_u = (b - l) * prob_next
            target_prob = toTensor(np.zeros(prob_next.size()))
            for i in range(target_prob.size(0)):
                target_prob[i].index_add_(0, l[i].long(), d_m_l[i])
                target_prob[i].index_add_(0, u[i].long(), d_m_u[i])
            
            _, log_prob = self.network(states)
            actions = toTensor(actions).long()
            log_prob = log_prob[self.batch_indices, actions, :]
            loss = -(target_prob * log_prob).sum(-1).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        
        if self.total_steps / config.rollout_length % config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
