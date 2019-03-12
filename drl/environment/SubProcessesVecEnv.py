from .VecEnv import VecEnv
import multiprocessing as mp
import numpy as np

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    
    def __init__(self, data):
        self.data = data
    
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.data)
    
    def __setstate__(self, ob):
        import pickle
        self.data = pickle.loads(ob)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                state, reward, done, info = env.step(data)
                if done:
                    state = env.reset()
                remote.send([state, reward, done, info])
            elif cmd == 'reset':
                state = env.reset()
                remote.send(state)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send([env.observation_space, env.action_space, env.spec])
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()

class SubProcessesVecEnv(VecEnv):
    def __init__(self, env_makers):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_makers)
        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe(True) for _ in range(self.num_envs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_maker)))
                   for work_remote, remote, env_maker in zip(self.work_remotes, self.remotes, env_makers)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        
        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
    
    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
    
    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]
    
    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
    
    def step_wait(self):
        self._assert_not_closed()
        data = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*data)
        return obs, np.asarray(rews), np.asarray(dones), infos
    
    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
    
    def __del__(self):
        if not self.closed:
            self.close()
