from collections import deque

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from .pendulum import PendulumEnv


class PendulumThetaEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, k=1):
        self.env = PendulumEnv()
        self.k = k

        self.observation_space = spaces.Box(low=-1, high=1, shape=(2*self.k,), dtype=np.float32)
        self.action_space = self.env.action_space

        self.buf = deque(maxlen=2*self.k)
        self._reset_buffer()

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __getattr__(self, item):
        return getattr(self.env, item)

    def _reset_buffer(self):
        for _ in range(2*self.k):
            self.buf.appendleft(0)

    def _update_buffer(self):
        th = self.env.get_theta()

        self.buf.appendleft(np.sin(th))
        self.buf.appendleft(np.cos(th))

        return np.asarray(self.buf.copy(), dtype=np.float32)

    def step(self,u):
        _, reward, done, info = self.env.step(u)
        return self._update_buffer(), reward, done, info

    def reset(self):
        self.env.reset()
        self._reset_buffer()
        return self._update_buffer()

    def render(self, mode='human'):
        return self.env.render(mode, False)
