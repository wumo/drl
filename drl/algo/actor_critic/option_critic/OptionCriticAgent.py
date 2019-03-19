from drl.algo.BaseAgent import BaseAgent
from drl.util.torch_utils import toNumpy, toTensor, epsilon_greedy
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

class OptionCriticAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        
        self.task = config.task_fn()
        self.states = config.state_normalizer(self.task.reset())
        
        self.q_options, self.betas, self.log_pi = self.network(self.states)
        self.options = epsilon_greedy(config.random_option_prob(config.num_workers), toNumpy(self.q_options))
        self.is_initial_betas = np.ones(self.config.num_workers)
        self.prev_options = np.copy(self.options)
    
    def step(self):
        config = self.config
        rollout = []
        
        q_options, betas, options, log_pi = self.q_options, self.betas, self.options, self.log_pi
        for _ in range(config.rollout_length):
            var_options = toTensor(options).long()
            worker_index = toTensor(np.arange(config.num_workers)).long()
            intra_log_pi = log_pi[worker_index, var_options, :]
            dist = torch.distributions.Categorical(intra_log_pi.exp())
            actions = dist.sample()
            
            next_states, rewards, terminals, infos = self.task.step(actions.cpu().detach().numpy().flatten())
            next_states = config.state_normalizer(next_states)
            rewards = config.reward_normalizer(rewards)
            q_options_next, betas_next, log_pi_next = self.network(next_states)
            rollout.append(
                [q_options, betas, options, self.prev_options, rewards, 1 - terminals, np.copy(self.is_initial_betas),
                 intra_log_pi, actions])
            self.is_initial_betas = np.asarray(terminals, dtype=np.float32)
            
            np_q_options_next = q_options_next.cpu().detach().numpy()
            np_betas_next = betas_next.gather(1, var_options.unsqueeze(1)).cpu().detach().numpy().flatten()
            options_next = np.copy(options)
            dice = np.random.rand(len(options_next))
            epsilon = config.random_option_prob(config.num_workers)
            for j in range(len(dice)):
                if dice[j] < np_betas_next[j] or terminals[j]:
                    options_next[j] = epsilon_greedy(epsilon, np_q_options_next[j])
            self.prev_options = options
            options = options_next
            q_options = q_options_next
            betas = betas_next
            log_pi = log_pi_next
            
            self.total_steps += config.num_workers
            if self.total_steps / config.num_workers % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
        
        self.options = options
        self.q_options = q_options
        self.betas = betas
        self.log_pi = log_pi
        
        target_q_options, _, _ = self.target_network(next_states)
        prev_options = toTensor(self.prev_options).long().unsqueeze(1)
        betas_prev_options = betas.gather(1, prev_options)
        
        returns = (1 - betas_prev_options) * target_q_options.gather(1, prev_options) + \
                  betas_prev_options * torch.max(target_q_options, dim=1, keepdim=True)[0]
        returns = returns.detach()
        
        processed_rollout = [None] * (len(rollout))
        for i in reversed(range(len(rollout))):
            q_options, betas, options, prev_options, rewards, terminals, is_initial_betas, log_pi, actions = rollout[i]
            options = toTensor(options).unsqueeze(1).long()
            prev_options = toTensor(prev_options).unsqueeze(1).long()
            terminals = toTensor(terminals).unsqueeze(1)
            rewards = toTensor(rewards).unsqueeze(1)
            is_initial_betas = toTensor(is_initial_betas).unsqueeze(1)
            returns = rewards + config.discount * terminals * returns
            
            q_omg = q_options.gather(1, options)
            log_action_prob = log_pi.gather(1, actions.unsqueeze(1))
            entropy_loss = (log_pi.exp() * log_pi).sum(-1).unsqueeze(1)
            
            q_prev_omg = q_options.gather(1, prev_options)
            v_prev_omg = torch.max(q_options, dim=1, keepdim=True)[0]
            advantage_omg = q_prev_omg - v_prev_omg
            advantage_omg.add_(config.termination_regularizer)
            betas = betas.gather(1, prev_options)
            betas = betas * (1 - is_initial_betas)
            processed_rollout[i] = [q_omg, returns, betas, advantage_omg.detach(), log_action_prob, entropy_loss]
        
        q_omg, returns, beta_omg, advantage_omg, log_action_prob, entropy_loss = map(lambda x: torch.cat(x, dim=0),
                                                                                     zip(*processed_rollout))
        pi_loss = -log_action_prob * (returns - q_omg.detach()) + config.entropy_weight * entropy_loss
        pi_loss = pi_loss.mean()
        q_loss = 0.5 * (q_omg - returns).pow(2).mean()
        beta_loss = (advantage_omg * beta_omg).mean()
        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
