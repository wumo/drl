import gym
from drl.environment.atari_wrappers import make_atari, wrap_deepmind, TransposeImage, FrameStack
import numpy as np
import matplotlib.pyplot as plt

env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env,
                    episode_life=True,
                    clip_rewards=True,
                    frame_stack=False,
                    scale=False)
obs_shape = env.observation_space.shape
if len(obs_shape) == 3:
    env = TransposeImage(env)
env = FrameStack(env, 4)

env.reset()
while True:
    obs,reward,done,info=env.step(env.action_space.sample())
    obs=np.asarray(obs)
    figure=plt.figure()
    for i in range(obs.shape[0]):
        a=figure.add_subplot(1,4,i+1)
        plt.gray()
        plt.imshow(obs[i])
    plt.show()
    if done:
        env.reset()
