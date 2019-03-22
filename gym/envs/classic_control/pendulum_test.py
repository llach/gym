import gym
import numpy as np
import cairocffi as cairo


class PendulumTestEnv(gym.Env):
    ''' For debugging pendulum env. in each episode, theta grows from 0 to 2*PI in N steps. '''

    def __init__(self, steps=20):

        # rendering things
        self.w , self.h = 64, 64
        self.surf = cairo.ImageSurface(cairo.FORMAT_RGB24, self.w, self.h)

        self.steps = steps
        self.thetas = np.linspace(0, 2*np.pi, self.steps)
        self.t = 0

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
        return self._render_pendulum()

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
