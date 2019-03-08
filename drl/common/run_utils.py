import datetime
import time
import numpy as np

def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and (agent.total_steps % config.save_interval == 0):
            agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
        if config.log_interval and (agent.total_steps % config.log_interval == 0) \
                and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            config.logger.info(f'total steps {agent.total_steps}, '
                               f'returns {np.mean(rewards):.2f}'
                               f'/{np.median(rewards):.2f}'
                               f'/{np.min(rewards):.2f}'
                               f'/{np.max(rewards)} '
                               f'(mean/median/min/max), '
                               f'{config.log_interval / (time.time() - t0):.2f} steps/s')
            t0 = time.time()
        if config.eval_interval and (agent.total_steps % config.eval_interval == 0):
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            break
        agent.step()
