import numpy as np
from anti_aliasing import AntiAliasingFilter
from mimic_filter import MimicFilter
from fourier_filter import PhasorEstimator


class Iec:
    def __init__(self, va, vb, vc, ia, ib, ic, t, sampling_period, b, c, md, R, XL, phasor_estimator_sampling_period, frequency=60):
        self.voltages = {
            'a': va,
            'b': vb,
            'c': vc
        }
        self.currents = {
            'a': ia,
            'b': ib,
            'c': ic
        }
        self.time = t
        self.sampling_period = sampling_period
        self.b = b
        self.c = c
        self.md = md
        self.R = R
        self.XL = XL
        self.phasor_estimator_sampling_period = phasor_estimator_sampling_period
        self.frequency = frequency
        self.apply_anti_aliasing_filter()
        self.resample()
        self.apply_mimic_filter()
        self.estimate_phasors()

    def apply_anti_aliasing_filter(self):
        for phase in self.voltages:
            aa_filter = AntiAliasingFilter(period=self.sampling_period, signal=self.voltages[phase],
                                           b=self.b, c=self.c)
            aa_filter.apply_filter()
            self.voltages[phase] = aa_filter.filtered_signal
        for phase in self.currents:
            aa_filter = AntiAliasingFilter(period=self.sampling_period, signal=self.currents[phase],
                                           b=self.b, c=self.c)
            aa_filter.apply_filter()
            self.currents[phase] = aa_filter.filtered_signal

    def resample(self):
        for phase in self.voltages:
            self.voltages[phase] = self.voltages[phase][::self.md].reshape(-1,)
        for phase in self.currents:
            self.currents[phase] = self.currents[phase][::self.md].reshape(-1,)
        self.time = self.time[::self.md].reshape(-1,)
        self.sampling_period = self.time[1] - self.time[0]

    def apply_mimic_filter(self):
        inductance = self.XL / (2 * np.pi * self.frequency)
        tau = inductance / self.R
        for phase in self.voltages:
            mimic_filter = MimicFilter(self.voltages[phase], tau, self.sampling_period)
            mimic_filter.apply_filter()
            self.voltages[phase] = mimic_filter.filtered_signal
        for phase in self.currents:
            mimic_filter = MimicFilter(self.currents[phase], tau, self.sampling_period)
            mimic_filter.apply_filter()
            self.currents[phase] = mimic_filter.filtered_signal

    def estimate_phasors(self):
        self.phasors = {
            'va': PhasorEstimator(self.voltages['a'], self.phasor_estimator_sampling_period),
            'vb': PhasorEstimator(self.voltages['b'], self.phasor_estimator_sampling_period),
            'vc': PhasorEstimator(self.voltages['c'], self.phasor_estimator_sampling_period),
            'ia': PhasorEstimator(self.currents['a'], self.phasor_estimator_sampling_period),
            'ib': PhasorEstimator(self.currents['b'], self.phasor_estimator_sampling_period),
            'ic': PhasorEstimator(self.currents['c'], self.phasor_estimator_sampling_period)
        }
        for phase in self.phasors:
            self.phasors[phase].estimate()
