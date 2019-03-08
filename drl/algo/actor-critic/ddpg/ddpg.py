import numpy as np
import time
import gym
import math
import torch
from drl.util.logx import EpochLogger
from drl.util.utils import toTensors
from drl.common.GAEBuffer import GAEBuffer
from drl.common.network import mlp_actor_critic

"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg(env_name, actor_critic=mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=1e-2,
         vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
         target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    logger = EpochLogger(**logger_kwargs)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    net = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    actor_optimizer = torch.optim.Adam(net.actor_params + net.shared_params, lr=pi_lr)
    critic_optimizer = torch.optim.Adam(net.critic_params + net.shared_params, lr=vf_lr)
    
    buf = GAEBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    
    def update():
        batch_obs, batch_acts, batch_adv, batch_ret, batch_logp_old = toTensors(buf.get())
        
        # VPG objectives
        def policy_loss(states, actions, advantages, log_prob_old):
            logp = net.actor(states, actions)
            ratio = (logp - log_prob_old).exp()  # pi(a|s) / pi_old(a|s)
            obj = ratio * advantages
            obj_clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            return -torch.min(obj, obj_clipped).mean(), (log_prob_old - logp).mean()
        
        def value_loss(states, returns):
            return 0.5 * (returns - net.critic(states)).pow(2).mean()
        
        pi_loss_old, _ = policy_loss(batch_obs, batch_acts, batch_adv, batch_logp_old)
        # Training policy network
        for i in range(train_pi_iters):
            pi_loss, kl = policy_loss(batch_obs, batch_acts, batch_adv, batch_logp_old)
            actor_optimizer.zero_grad()
            pi_loss.backward()
            actor_optimizer.step()
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        # Training value network
        v_loss_old = value_loss(batch_obs, batch_ret)
        for _ in range(train_v_iters):
            # VPG objectives
            v_loss = value_loss(batch_obs, batch_ret)
            critic_optimizer.zero_grad()
            v_loss.backward()
            critic_optimizer.step()
        
        # Log changes from update
        
        # Info (useful to watch during learning)
        pi_loss_new, kl = policy_loss(batch_obs, batch_acts, batch_adv, batch_logp_old)
        v_loss_new = value_loss(batch_obs, batch_ret)
        logp = net.actor(batch_obs, batch_acts)
        approx_entropy = (-logp).mean()
        ratio = (logp - batch_logp_old).exp()
        clipped = ((ratio > (1 + clip_ratio)) + (ratio < (1 - clip_ratio))).clamp(0, 1)
        clipfrac = clipped.float().mean()
        logger.store(LossPi=pi_loss_old.item(), LossV=v_loss_old.item(),
                     KL=kl.item(), Entropy=approx_entropy.item(),
                     ClipFrac=clipfrac.item(),
                     DeltaLossPi=(pi_loss_new - pi_loss_old).item(),
                     DeltaLossV=(v_loss_new - v_loss_old).item())
    
    start_time = time.time()
    obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    
    max_EpRet = -math.inf
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            action, v_t, log_prob_t = net(obs)
            
            # save
            buf.store(obs, action.tolist(), rew, v_t.item(), log_prob_t.tolist())
            logger.store(Value=v_t.item())
            
            obs, rew, done, _ = env.step(action.tolist())
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
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        
        mean, std = logger.get_stats('EpRet')
        if mean > max_EpRet:
            # save model
            print(f'Saving model due to improvement {mean} > {max_EpRet}')
            max_EpRet = mean
            torch.save({
                'env': env_name,
                'ac_kwargs': ac_kwargs,
                'model': net.state_dict()}, "model.pth")
        # Perform VPG update!
        update()
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('Value', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
    
    # torch.save(net.state_dict(), "./trained_model/ppo.pt")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()
    
    ddpg(args.env, actor_critic=mlp_actor_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
