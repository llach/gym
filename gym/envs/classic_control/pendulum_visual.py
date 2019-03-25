import cv2
import gym
import cairocffi as cairo
from skimage.transform import resize

from gym import spaces
from gym.utils import seeding
import numpy as np


class PendulumVisualEnv(gym.Env):

    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05

        # rendering things
        self.w, self.h = 64, 64
        self.surf = cairo.ImageSurface(cairo.FORMAT_RGB24, self.w, self.h)

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(64, 64, 1), dtype=np.float32)
        self.frame_t = None
        self.has_window = False
        self.costs = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        self.costs = float(costs)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def render(self, mode=''):
        if self.frame_t is None: return

        if not self.has_window:
            cv2.namedWindow('pendulum')

        img = resize(self.frame_t.copy(), [512, 512], mode='reflect', anti_aliasing=True)
        cv2.putText(img, 'theta {:.3}'.format(angle_normalize(self._get_theta())), (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, .5, 2)

        cv2.putText(img, 'costs {:.3}'.format(self.costs), (250, 30),
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
        cr.rotate(np.pi-self.state[0])

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

    def _get_theta(self):
        return float(self.state[0])

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
