import cv2
import gym
import cairocffi as cairo
from skimage.transform import resize

from gym import spaces
from gym.utils import seeding
import numpy as np
from .pendulum import PendulumEnv


class PendulumVisualEnv(gym.Env):

    def __init__(self):
        self.env = PendulumEnv()

        # rendering things
        self.w, self.h = 64, 64
        self.surf = cairo.ImageSurface(cairo.FORMAT_RGB24, self.w, self.h)

        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=1, shape=(64, 64, 1), dtype=np.float32)

        self.frame_t = None
        self.has_window = False
        self.reward = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # pendulum env calculates new th & thdot given force u
        _, reward, done, info = self.env.step(u)

        # we save th
        self.th = self.env.get_theta()
        self.reward = reward

        # render current pendulum based on th_t
        self.frame_t = self._render_pendulum()

        return self.frame_t.copy(), reward, done, info

    def reset(self):
        self.env.reset()

        # we save th
        self.th = self.env.get_theta()

        # render current pendulum based on th_t
        self.frame_t = self._render_pendulum()

        return self.frame_t.copy()

    def render(self, mode=''):
        if self.frame_t is None: return

        if not self.has_window:
            cv2.namedWindow('pendulum')

        img = resize(self.frame_t.copy(), [512, 512], mode='reflect', anti_aliasing=True)
        cv2.putText(img, 'theta {:.3}'.format(angle_normalize(self.th)), (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, .5, 2)

        cv2.putText(img, 'costs {:.3}'.format(float(self.reward)), (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, .5, 2)

        cv2.imshow('pendulum', img)
        cv2.waitKey(1)

    def _render_pendulum(self):
        cr = cairo.Context(self.surf)

        # draw background
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        # apply transforms
        cr.translate((self.w / 2), self.h / 2)
        cr.rotate(np.pi-self.th)

        # draw shapes that form the capsule
        cr.rectangle(-2.5, 0, 5, 27)
        cr.arc(0, 0, 2.5, 0, 2 * np.pi)
        cr.arc(0, (self.h / 2) - 4, 2.5, 0, 2 * np.pi)

        # draw color
        cr.set_source_rgb(.8, .3, .3)
        cr.fill()

        # center sphere
        cr.arc(0, 0, 1, 0, 2 * np.pi)
        cr.set_source_rgb(0, 0, 0)
        cr.fill()

        # reshape, delete fourth (alpha) channel, greyscale and normalise
        return np.expand_dims(np.dot(np.frombuffer(self.surf.get_data(), np.uint8).reshape([self.w, self.h, 4])[..., :3], [0.299, 0.587, 0.114]), -1)/255

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
