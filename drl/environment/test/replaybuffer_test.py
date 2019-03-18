import gym
from drl.environment.atari_wrappers import make_atari, wrap_deepmind, FrameStack
from drl.common.ReplayBuffer import ReplayBuffer
import numpy as np
import pygame
from pygame import display, time, surfarray

def grayscale(img):
    return np.stack((img,) * 3, axis=-1)

def draw(screen, img, xoffset, yoffset, width, height):
    img = surfarray.make_surface(grayscale(img))
    img = pygame.transform.scale(img, (width, height))
    screen.blit(img, (xoffset, yoffset))

if __name__ == '__main__':
    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(None, 36)
    
    clock = time.Clock()
    FPS = 60
    running = True
    frame = 84
    k = 4
    batch_size = 32
    scale = 1
    h_max = 8
    w = frame * k * scale * (batch_size // h_max)
    h = frame * scale * h_max
    
    screen = display.set_mode((w, h))
    
    env = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env,
                        episode_life=True,
                        clip_rewards=False,
                        frame_stack=False,
                        scale=False)
    obs_shape = env.observation_space.shape
    env = FrameStack(env, k)
    
    replay = ReplayBuffer(env, memory_size=int(1e6), batch_size=batch_size, stack=k)
    
    state = env.reset()
    while running:
        # clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        display.set_caption(f'{clock.get_fps():.2f}')
        
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done: next_state = env.reset()
        replay.store([state, action, reward, next_state, done])
        if replay.size < replay.memory_size:
            for _ in range(replay.memory_size):
                replay.store([state, action, reward, next_state, done])
        state = next_state
        print(replay.size)
        screen.fill((255, 255, 255))
        if replay.size > batch_size:
            states, actions, rewards, next_states, terminals = replay.sample(batch_size)
            
            states = np.reshape(states, (batch_size // h_max, h_max,) + states.shape[1:])
            xzero = 0
            for column in states:
                yoffset = 0
                for row in column:
                    xoffset = xzero
                    for j in row:
                        draw(screen, j, xoffset, yoffset, frame * scale, frame * scale)
                        xoffset += frame * scale
                    yoffset += frame * scale
                xzero += k * frame * scale
        
        display.flip()
