import numpy as np
import gym
import torch
import torch.nn  as nn
from torch.distributions.categorical import Categorical
from drl.util.utils import tensor, tensors, layer_init, DEVICE
import drl.vpg.core as core

class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.num_episodes = 0
    
    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
    
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
        
        self.num_episodes += 1
    
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        num_episodes = self.num_episodes
        self.ptr, self.path_start_idx, self.num_episodes = 0, 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = core.statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf, num_episodes]

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim
    
    def forward(self, x):
        return x

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(32,), activation=torch.tanh):
        super(FCBody, self).__init__()
        
        dims = list(state_dim) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.activation = activation
        self.feature_dim = dims[-1]
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

class CategoricalActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, shared_body=None, actor_body=None, critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        if shared_body is None: shared_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(shared_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(shared_body.feature_dim)
        self.shared_body = shared_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim))
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1))
        
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.shared_params = list(self.shared_body.parameters())
        
        self.to(DEVICE)
    
    def forward(self, obs):
        obs = tensor(obs)
        shared = self.shared_body(obs)
        
        action, log_prob = self.actor(obs, None, shared)
        v = self.critic(obs, shared)
        return action, v, log_prob
    
    def actor(self, obs, action, shared=None):
        if shared is None: shared = self.shared_body(tensor(obs))
        action_logits = self.fc_action(self.actor_body(shared))
        action_dist = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            action = action_dist.sample()
            return action, action_dist.log_prob(action)
        else:
            return action_dist.log_prob(action)
    
    def critic(self, obs, shared=None):
        if shared is None: shared = self.shared_body(tensor(obs))
        return self.fc_critic(self.critic_body(shared))

def mlp_actor_critic(observation_space, action_space, hidden_sizes=(32,), activation=torch.tanh):
    assert isinstance(action_space, gym.spaces.discrete.Discrete), "Currently only support Discrete."
    
    net = CategoricalActorCriticNet(observation_space.shape, action_space.n,
                                    actor_body=FCBody(observation_space.shape, hidden_sizes, activation=activation),
                                    critic_body=FCBody(observation_space.shape, hidden_sizes, activation=activation))
    
    return net

def vpg(env_fn, actor_critic=mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=50, gamma=0.99, pi_lr=1e-2,
        vf_lr=1e-2, train_v_iters=80, lam=0.97, max_ep_len=1000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    net = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    actor_optimizer = torch.optim.Adam(net.actor_params + net.shared_params, lr=pi_lr)
    critic_optimizer = torch.optim.Adam(net.critic_params + net.shared_params, lr=vf_lr)
    
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    
    def update():
        batch_obs, batch_acts, batch_adv, batch_ret, batch_logp_old, num_episodes = tensors(buf.get())
        
        # VPG objectives
        def policy_loss(states, actions, advantages, num_episodes):
            log_prob = net.actor(states, actions)
            return -(log_prob * advantages).sum() / num_episodes
        
        def value_loss(states, returns, num_episodes):
            return 0.5 * (returns - net.critic(states)).pow(2).sum() / num_episodes
        
        pi_loss = policy_loss(batch_obs, batch_acts, batch_adv, num_episodes)
        actor_optimizer.zero_grad()
        pi_loss.backward()
        actor_optimizer.step()
        
        v_loss_old = value_loss(batch_obs, batch_ret, num_episodes)
        for _ in range(train_v_iters):
            # VPG objectives
            v_loss = value_loss(batch_obs, batch_ret, num_episodes)
            critic_optimizer.zero_grad()
            v_loss.backward()
            critic_optimizer.step()
        
        # Log changes from update
        
        # Info (useful to watch during learning)
        logp = net.actor(batch_obs, batch_acts)
        
        pi_loss_new = policy_loss(batch_obs, batch_acts, batch_adv, num_episodes)
        v_loss_new = value_loss(batch_obs, batch_ret, num_episodes)
        approx_kl = (batch_logp_old - logp).mean()
        approx_entropy = (-logp).mean()
        print(f"numEpisodes={num_episodes}"
              f"lossPi={pi_loss},lossV={v_loss_old},"
              f"kl={approx_kl},entropy={approx_entropy},"
              f"deltaLossPi={pi_loss_new - pi_loss},deltaLossV={v_loss_new - v_loss_old}")
    
    obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        EpRet = []
        EpLen = []
        for t in range(steps_per_epoch):
            action, v_t, log_prob_t = net(obs)
            
            # save
            buf.store(obs, action.item(), rew, v_t.item(), log_prob_t.item())
            
            obs, rew, done, _ = env.step(action.item())
            ep_ret += rew
            ep_len += 1
            
            terminal = done or (ep_len == max_ep_len)
            if terminal or (t == steps_per_epoch - 1):
                if not terminal:
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps.')
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = rew if done else net.critic(obs).item()
                buf.finish_path(last_val)
                if terminal:
                    EpRet.append(ep_ret)
                    EpLen.append(ep_len)
                obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        # Perform VPG update!
        update()
        print(f"{np.mean(EpRet)},{np.mean(EpLen)}")
    
    torch.save(net.state_dict(), "./trained_model/vpg.pt")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=32)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()
    
    vpg(lambda: gym.make(args.env), actor_critic=mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
