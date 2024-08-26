import numpy as np


class AntiAliasing():
    def __init__(self, period, signal, b, c):
        self.signal = signal
        self.period = period
        self.a = 2 / self.period
        self.b = b
        self.c = c
        self.filter()

    def filter(self):
        c0 = self.c / (self.a**2 + self.a*self.b + self.c)
        c1 = 2 * c0
        c2 = c0
        d1 = 2 * (self.c - self.a**2) / (self.a**2 + self.a*self.b + self.c)
        d2 = (self.a**2 + self.c - self.a*self.b) / (self.a**2 + self.a*self.b + self.c)

        num_samples = len(self.signal)
        self.x_out = np.zeros((num_samples, 1))

        self.x_out[0] = c0 * self.signal[0]
        self.x_out[1] = c0 * self.signal[1] + c1 * self.signal[0] - d1 * self.x_out[0]

        for i in range(2, num_samples):
            self.x_out[i] = c0 * self.signal[i-2] + c1 * self.signal[i-1] + c2 * self.signal[i] - d1 * self.x_out[i-1] - d2 * self.x_out[i-2]
