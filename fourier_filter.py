import numpy as np


class FourierFilter:
    def __init__(self, sample_rate, is_full_period=True):
        self.sample_rate = sample_rate
        self.is_full_period = is_full_period
        self.create_filter()

    def create_filter(self):
        if self.is_full_period:
            filter_args = np.linspace((2*np.pi / self.sample_rate), 2*np.pi, self.sample_rate)
            self.cosine_filter = ((2 / self.sample_rate) * np.cos(filter_args))
            self.sine_filter = ((2 / self.sample_rate) * np.sin(filter_args))
        else:
            filter_args = np.linspace((2*np.pi / self.sample_rate), np.pi, (self.sample_rate / 2))
            self.cosine_filter = ((4 / self.sample_rate) * np.cos(filter_args))
            self.sine_filter = ((4 / self.sample_rate) * np.sin(filter_args))


class PhasorEstimator:
    def __init__(self, signal, sample_rate, is_full_period=True):
        self.signal = signal
        self.fourier_filter = FourierFilter(sample_rate, is_full_period)

    def estimate(self):
        self.real_part = np.convolve(self.signal, self.fourier_filter.cosine_filter)
        self.imaginary_part = np.convolve(self.signal, self.fourier_filter.sine_filter)
        self.amplitude = np.sqrt(self.real_part**2 + self.imaginary_part**2)
        phase_rad = np.arctan2(self.imaginary_part, self.real_part)
        self.phase = np.degrees(phase_rad)
