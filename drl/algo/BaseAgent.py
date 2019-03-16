import torch
import numpy as np
import time
import datetime
import os
import psutil
import gc
import json
from drl.util.logger import pretty_time_delta, pretty_memory
from drl.util.serialization import toJson

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.total_steps = 0
        self.episode_rewards = []
        self.network = None
        self.task = None
    
    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
    
    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
    
    def eval_step(self, state):
        pass
    
    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        total_rewards = 0
        while True:
            action = self.eval_step(state)
            state, reward, done, _ = env.step([action])
            total_rewards += reward[0]
            if done[0]:
                break
        return total_rewards
    
    def eval_episodes(self):
        rewards = []
        for ep in range(self.config.eval_episodes):
            rewards.append(self.eval_episode())
        mean_rewards, std = np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))
        self.config.logger.info(f'evaluation episode return: {mean_rewards}'
                                f'({std})')
        self.config.logger.add_scalar("eval_mean_rewards", mean_rewards, self.total_steps)
    
    def step(self):
        raise NotImplementedError
    
    def close(self):
        if self.task is not None:
            self.task.close()
    
    def run_steps(self):
        config = self.config
        output = json.dumps(toJson(config, 1), separators=(',', ':\t'), indent=4, sort_keys=True)
        print(output)
        agent_name = self.__class__.__name__
        t0 = time.time()
        mean_steps_per_s = 0
        N = 0
        last_log_steps = 0
        last_eval_steps = 0
        last_gc_steps = 0
        process = psutil.Process(os.getpid())
        while True:
            if config.save_interval and (self.total_steps % config.save_interval == 0):
                self.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
            if config.log_interval and (self.total_steps - last_log_steps > config.log_interval) \
                    and len(self.episode_rewards):
                rewards = self.episode_rewards
                self.episode_rewards = []
                mean_rewards, media_rewards, min_rewards, max_rewards = np.mean(rewards), np.median(
                    rewards), np.min(rewards), np.max(rewards)
                steps_per_s = (self.total_steps - last_log_steps) / (time.time() - t0)
                last_log_steps = self.total_steps
                mean_steps_per_s = (mean_steps_per_s * N + steps_per_s) / (N + 1)
                N += 1
                ETA = (config.max_steps - self.total_steps) / mean_steps_per_s
                mem = psutil.virtual_memory()
                percent = str(mem.percent)
                used_mem = process.memory_info().rss
                config.logger.add_scalar("mean_rewards", mean_rewards, self.total_steps)
                config.logger.info(f'total steps {self.total_steps}, '
                                   f'returns {mean_rewards:.2f}'
                                   f'/{media_rewards:.2f}'
                                   f'/{min_rewards:.2f}'
                                   f'/{max_rewards:.2f} '
                                   f'(mean/median/min/max), '
                                   f'{steps_per_s:.0f} steps/s, '
                                   f'ETA: {pretty_time_delta(ETA)}, '
                                   f'mem:{pretty_memory(used_mem)}({percent}%)')
                t0 = time.time()
            if config.eval_interval and (self.total_steps - last_eval_steps > config.eval_interval):
                last_eval_steps = self.total_steps
                self.eval_episodes()
            if config.gc_interval and (self.total_steps-last_gc_steps>config.gc_interval):
                last_gc_steps=self.total_steps
                before_gc_used_mem = process.memory_info().rss
                gc.collect()
                after_gc_used_mem = process.memory_info().rss
                config.logger.info(f'gc: before={pretty_memory(before_gc_used_mem)}; '
                                   f'after={pretty_memory(after_gc_used_mem)}')
            if config.max_steps and self.total_steps >= config.max_steps:
                break
            self.step()
        config.eval_env.close()
        self.close()
