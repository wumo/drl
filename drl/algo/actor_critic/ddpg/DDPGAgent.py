from drl.algo.BaseAgent import BaseAgent
from drl.util.torch_utils import toNumpy, range_tensor, toTensor
import numpy as np

class DDPGAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        
        self.task = config.task_fn()
        self.random_process.reset_states()
        self.states = config.state_normalizer(self.task.reset())
    
    def step(self):
        config = self.config
        actions = self.network(self.states)
        actions = toNumpy(actions)
        actions += self.random_process.sample()
        next_states, rewards, dones, _ = self.task.step(actions)
        next_states = self.config.state_normalizer(next_states)
        rewards = self.config.reward_normalizer(rewards)
        self.replay.store([self.states, actions, rewards, next_states,
                           dones.astype(np.uint8)])
        if dones[0]:
            self.random_process.reset_states()
        self.states = next_states
        self.total_steps += 1
        
        if self.replay.size >= config.min_memory_size:
            experiences = self.replay.sample(config.batch_size)
            states, actions, rewards, next_states, terminals = experiences
            states = states.squeeze(1)
            actions = actions.squeeze(1)
            rewards = toTensor(rewards)
            next_states = next_states.squeeze(1)
            terminals = toTensor(terminals)
            
            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, toTensor(actions))
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
            
            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()
            
            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()
            
            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()
            
            # soft_update
            for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
                target_param.detach_()
                target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                   param * self.config.target_network_mix)
