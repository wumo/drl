import gym
from drl.environment.atari_wrappers import make_atari, wrap_deepmind, FrameStack
import numpy as np
import pygame
from pygame import display, time, surfarray

def grayscale(img):
    return np.stack((img,) * 3, axis=-1)

pygame.init()
pygame.font.init()
font = pygame.font.Font(None, 36)

clock = time.Clock()
FPS = 60
running = True
frame = 84
k = 4
scale = 4
w = frame * k * scale
h = frame * scale
screen = display.set_mode((w, h))

env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env,
                    episode_life=True,
                    clip_rewards=False,
                    frame_stack=False,
                    scale=False)
obs_shape = env.observation_space.shape
env = FrameStack(env, 4)

obs = env.reset()
while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    display.set_caption(f'{clock.get_fps():.2f}')
    
    obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        obs = env.reset()
    obs = np.asarray(obs)
    xoffset = 0
    for img in obs:
        img = surfarray.make_surface(grayscale(img))
        img = pygame.transform.scale(img, (frame * scale, frame * scale))
        screen.blit(img, (xoffset, 0))
        xoffset += frame * scale
    # surfarray.blit_array(screen, obs[-1])
    
    # screen.fill((255, 255, 255))
    # text = font.render("fps", 1, (10, 10, 10))
    # textpos = text.get_rect(centerx=screen.get_width() / 2)
    # screen.blit(text, textpos)
    
    display.flip()
    # time.delay(100)
