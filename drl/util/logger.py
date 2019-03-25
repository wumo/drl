#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from tensorboardX import SummaryWriter
from drl.util.misc import mkdir
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s')
from .misc import *

def pretty_memory(mem):
    if mem < 1024:
        return f'{mem}B'
    elif mem / 1024 < 1024:
        return f'{mem / 1024:.2f}KB'
    elif mem / 1024 / 1024 < 1024:
        return f'{mem / 1024 / 1024:.2f}MB'
    else:
        return f'{mem / 1024 / 1024 / 1024:.2f}GB'

def pretty_time_delta(seconds):
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm%ds' % (sign_string, minutes, seconds)
    else:
        return '%s%ds' % (sign_string, seconds)

def get_logger(tag=None, skip=False, level=logging.INFO):
    log_dir = str(Path.home())
    logger = logging.getLogger()
    logger.setLevel(level)
    if tag is not None:
        mkdir(f'{log_dir}/log')
        fh = logging.FileHandler(f'{log_dir}/log/%s-%s.txt' % (tag, get_time_str()))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        fh.setLevel(level)
        logger.addHandler(fh)
    return Logger(logger, f'{log_dir}/tf_log/{tag}-{get_time_str()}', skip)

class Logger(object):
    def __init__(self, vanilla_logger, log_dir, skip=False):
        if not skip:
            self.writer = SummaryWriter(log_dir)
        if vanilla_logger is not None:
            self.info = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning
        self.skip = skip
        self.all_steps = {}
    
    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v
    
    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step
    
    def add_scalar(self, tag, value, step=None):
        if self.skip:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)
    
    def add_histogram(self, tag, values, step=None):
        if self.skip:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)

def experiment_count():
    home = str(Path.home())
    mkdir(f'{home}/.experiment')
    filepath = f'{home}/.experiment/count'
    count = 1
    if os.path.exists(filepath):
        with open(filepath, 'r+') as f:
            try:
                count = int(f.read()) + 1
            except:
                pass
    with open(filepath, 'w+') as f:
        f.write(str(count))
    return count
