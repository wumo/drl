import gym
import torch
from drl.common.network import mlp_actor_critic

if __name__ == '__main__':
    saved = torch.load('vpg.pth')
    env = gym.make(saved['env'])
    state_dict = saved['model']
    ac_kwargs = saved['ac_kwargs']
    net = mlp_actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    net.load_state_dict(state_dict)
    net.eval()
    
    obs = env.reset()
    
    while True:
        a, _ = net.actor(obs)
        obs, rew, done, _ = env.step(a.item())
        env.render()
        if done:
            obs = env.reset()
