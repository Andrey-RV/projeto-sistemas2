import numpy as np
from anti_aliasing import AntiAliasingFilter
from mimic_filter import MimicFilter
from fourier_filter import PhasorEstimator


class Iec:
    def __init__(self, va, vb, vc, ia, ib, ic, t, sampling_period, b, c, md,
                 R, XL, phasor_estimator_samples_per_cycle, frequency=60):
        self.signals = {
            'va': va,
            'vb': vb,
            'vc': vc,
            'ia': ia,
            'ib': ib,
            'ic': ic
        }
        self.time = t
        self.sampling_period = sampling_period
        self.b = b
        self.c = c
        self.md = int(md)
        self.R = R
        self.XL = XL
        self.phasor_estimator_samples_per_cycle = phasor_estimator_samples_per_cycle
        self.frequency = frequency
        self.apply_anti_aliasing_filter()
        self.resample()
        self.apply_mimic_filter()
        self.estimate_phasors()

    def apply_anti_aliasing_filter(self):
        for signal in self.signals:
            anti_aliasing_filter = AntiAliasingFilter(self.sampling_period, self.signals[signal], self.b, self.c)
            anti_aliasing_filter.apply_filter()
            self.signals[signal] = anti_aliasing_filter.filtered_signal
        print('Filtro antialiasing aplicado.')

    def resample(self):
        for signal in self.signals:
            self.signals[signal] = self.signals[signal][::self.md].reshape(-1)
        self.time = np.array(self.time[::self.md]).reshape(-1)
        self.sampling_period = self.time[1] - self.time[0]
        print(f'Sinal reamostrado com fator de subamostragem = {self.md}.')

    def apply_mimic_filter(self):
        inductance = self.XL / (2 * np.pi * self.frequency)
        tau = (inductance / self.R) / self.sampling_period
        for signal in self.signals:
            mimic_filter = MimicFilter(self.signals[signal], tau, self.sampling_period)
            mimic_filter.apply_filter()
            self.signals[signal] = mimic_filter.filtered_signal
        print('Filtro mimético aplicado.')

    def estimate_phasors(self):
        self.phasors = {
            'va': PhasorEstimator(self.signals['va'], self.phasor_estimator_samples_per_cycle),
            'vb': PhasorEstimator(self.signals['vb'], self.phasor_estimator_samples_per_cycle),
            'vc': PhasorEstimator(self.signals['vc'], self.phasor_estimator_samples_per_cycle),
            'ia': PhasorEstimator(self.signals['ia'], self.phasor_estimator_samples_per_cycle),
            'ib': PhasorEstimator(self.signals['ib'], self.phasor_estimator_samples_per_cycle),
            'ic': PhasorEstimator(self.signals['ic'], self.phasor_estimator_samples_per_cycle)
        }
        for signal in self.phasors:
            self.phasors[signal].estimate()
            self.phasors[signal].amplitude = self.phasors[signal].amplitude[:len(self.time)]
            self.phasors[signal].phase = self.phasors[signal].phase[:len(self.time)]
        print('Fasores estimados.')
