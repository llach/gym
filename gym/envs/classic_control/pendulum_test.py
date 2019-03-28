import gym
import cv2
import numpy as np
import cairocffi as cairo

from skimage.transform import resize

from gym import spaces
from gym.utils import seeding
from .pendulum import PendulumEnv

class PendulumTestEnv(gym.Env):
    ''' For debugging pendulum env. in each episode, theta grows from 0 to 2*PI in N steps. '''

    def __init__(self, steps=20):
        self.env = PendulumEnv()

        # rendering things
        self.w , self.h = 64, 64
        self.surf = cairo.ImageSurface(cairo.FORMAT_RGB24, self.w, self.h)

        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=1, shape=(64, 64, 1), dtype=np.float32)

        self.steps = steps
        self.thetas = np.linspace(0, 2*np.pi, self.steps)
        self.t = 0

        self.frame_t = None
        self.has_window = False

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_theta(self):
        return self.thetas[self.t]

    def step(self, a):

        if self.t < self.steps-1:
            self.t += 1
            d = False
        else:
            self.t = 0
            d = True

        return self._get_obs(), 0, d, {}

    def reset(self):
        self.t = 0
        return self._get_obs()

    def render(self, mode=''):
        if self.frame_t is None: return

        if not self.has_window:
            cv2.namedWindow('pendulum')

        img = resize(self.frame_t.copy(), [512, 512], mode='reflect', anti_aliasing=True)
        cv2.putText(img, 'theta {:.3}'.format(angle_normalize(self._get_theta())), (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, .5, 2)

        cv2.putText(img, '~costs {:.3}'.format(angle_normalize(self._get_theta())**2), (250, 30),
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
        cr.rotate(np.pi-self._get_theta())

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

    def _get_obs(self):
        self.frame_t = self._render_pendulum()
        return self.frame_t


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
