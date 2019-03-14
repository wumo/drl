import time
import datetime
import numpy as np
import json
from drl.util.serialization import toJson

def run_steps(agent):
    config = agent.config
    output = json.dumps(toJson(config, 1), separators=(',', ':\t'), indent=4, sort_keys=True)
    print(output)
    agent_name = agent.__class__.__name__
    t0 = time.time()
    mean_steps_per_s = 0
    N = 0
    last_steps = 0
    while True:
        if config.save_interval and (agent.total_steps % config.save_interval == 0):
            agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
        if config.log_interval and (agent.total_steps - last_steps > config.log_interval) \
                and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            mean_rewards, media_rewards, min_rewards, max_rewards = np.mean(rewards), np.median(
                rewards), np.min(rewards), np.max(rewards)
            steps_per_s = (agent.total_steps - last_steps) / (time.time() - t0)
            last_steps = agent.total_steps
            mean_steps_per_s = (mean_steps_per_s * N + steps_per_s) / (N + 1)
            N += 1
            ETA = (config.max_steps - agent.total_steps) / mean_steps_per_s
            config.logger.add_scalar("mean_rewards", mean_rewards, agent.total_steps)
            config.logger.info(f'total steps {agent.total_steps}, '
                               f'returns {mean_rewards:.2f}'
                               f'/{media_rewards:.2f}'
                               f'/{min_rewards:.2f}'
                               f'/{max_rewards} '
                               f'(mean/median/min/max), '
                               f'{steps_per_s:.2f} steps/s'
                               f' ETA: {str(datetime.timedelta(seconds=ETA))}')
            t0 = time.time()
        if config.eval_interval and (agent.total_steps % config.eval_interval == 0):
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            break
        agent.step()
