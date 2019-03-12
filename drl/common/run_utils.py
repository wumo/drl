import datetime
import time
import numpy as np
import json
from drl.util.serialization import toJson
from drl.util.plot import plot
import logging
import matplotlib.pyplot as plt

def run_steps(agent):
    config = agent.config
    output = json.dumps(toJson(config, 1), separators=(',', ':\t'), indent=4, sort_keys=True)
    print(output)
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and (agent.total_steps % config.save_interval == 0):
            agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
        if config.log_interval and (agent.total_steps % config.log_interval == 0) \
                and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            mean_rewards, media_rewards, min_rewards, max_rewards, steps_per_s = np.mean(rewards), np.median(
                rewards), np.min(rewards), np.max(rewards), config.log_interval / (time.time() - t0)
            config.logger.add_scalar("mean_rewards", mean_rewards, agent.total_steps)
            config.logger.info(f'total steps {agent.total_steps}, '
                               f'returns {mean_rewards:.2f}'
                               f'/{media_rewards:.2f}'
                               f'/{min_rewards:.2f}'
                               f'/{max_rewards} '
                               f'(mean/median/min/max), '
                               f'{steps_per_s:.2f} steps/s')
            t0 = time.time()
        if config.eval_interval and (agent.total_steps % config.eval_interval == 0):
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            break
        agent.step()
