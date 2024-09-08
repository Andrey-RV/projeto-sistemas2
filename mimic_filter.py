import numpy as np


class MimicFilter:
    def __init__(self, signal, tau, sample_period):
        self.signal = signal
        self.tau = tau
        self.sample_period = sample_period

    def apply_filter(self):
        num_samples = len(self.signal)
        self.k = 1 / np.sqrt((1 + self.tau - self.tau * np.cos(120*np.pi * self.sample_period))**2 +
                             (self.tau * np.sin(120*np.pi * self.sample_period))**2)

        self.filtered_signal = np.zeros((num_samples, 1))

        self.filtered_signal[0] = self.k * (1 + self.tau) * self.signal[0]
        for i in range(1, num_samples):
            self.filtered_signal[i] = self.k * ((1 + self.tau) * self.signal[i] - self.tau * self.signal[i-1])
