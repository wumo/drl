from drl.algo.critic.dqn.dqn_example import dqn_pixel_atari
from drl.algo.critic.ndqn.nstepdqn_example import nstepdqn_pixel_atari
from drl.algo.critic.c51.categorical_dqn_example import categorical_dqn_pixel_atari
from drl.algo.actor.a2c.a2c_example import a2c_pixel_atari
from drl.algo.actor.ppo.ppo_example import ppo_pixel_atari
from drl.util.torch_utils import random_seed, select_device
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', nargs='+',
                        default=['a2c_pixel_atari', 'ppo_pixel_atari', 'nstepdqn_pixel_atari',
                                 'dqn_pixel_atari', 'categorical_dqn_pixel_atari'])
    parser.add_argument('--env', nargs='+', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--tag', default='bench')
    args = parser.parse_args()
    tag = args.tag
    games = args.env
    algos = args.algo
    
    print(f'games: {games}')
    print(f'algorithms: {[algo for algo in algos]}')
    
    random_seed()
    select_device(0)
    for game in games:
        for algo in algos:
            algo = globals()[algo]
            algo(game, tag)
