import numpy as np


class FourierFilter:
    def __init__(self, sample_rate, is_full_period=True):
        self.sample_rate = sample_rate
        self.is_full_period = is_full_period
        self.create_filter()

    def create_filter(self):
        if self.is_full_period:
            filter_args = np.linspace((2*np.pi / self.sample_rate), 2*np.pi, self.sample_rate)
            self.cosine_filter = ((2 / self.sample_rate) * np.cos(filter_args)).round(4)
            self.sine_filter = ((2 / self.sample_rate) * np.sin(filter_args)).round(4)
        else:
            filter_args = np.linspace((2*np.pi / self.sample_rate), np.pi, (self.sample_rate / 2))
            self.cosine_filter = ((4 / self.sample_rate) * np.cos(filter_args)).round(4)
            self.sine_filter = ((4 / self.sample_rate) * np.sin(filter_args)).round(4)


class PhasorEstimator:
    def __init__(self, signal, sample_rate, is_full_period=True):
        self.signal = signal
        self.sample_rate = sample_rate
        self.is_full_period = is_full_period
        self.fourier_filter = FourierFilter(sample_rate, is_full_period)

    def estimate(self):
        self.real_part = np.convolve(self.signal, self.fourier_filter.cosine_filter).round(4)
        self.imaginary_part = np.convolve(self.signal, self.fourier_filter.sine_filter).round(4)
        self.amplitude = np.sqrt(self.real_part**2 + self.imaginary_part**2).round(4)
        phase_rad = np.arctan2(self.imaginary_part, self.real_part)
        self.phase = np.degrees(phase_rad).round(4)


vc = np.array([-9.0745e1, 3.4725e0, 9.0736e1, -3.7006e0, -9.0727e1, 3.9286e0, 9.0717e1, -2.6706e0])
phasor = PhasorEstimator(signal=vc, sample_rate=4, is_full_period=True)
phasor.estimate()
print(f"Real Part: {phasor.real_part}")
print(f"Imaginary Part: {phasor.imaginary_part}")
print(f"Amplitude: {phasor.amplitude}")
print(f"Phase: {phasor.phase}")
