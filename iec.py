import numpy as np
from filters import AntiAliasingFilter, FourierFilter, MimicFilter


class PhasorEstimator:
    def __init__(self, signal, samples_per_cycle, num_points):
        '''
        Instancia um estimador de fasor.
        Attributes:
            signal: Uma sequência de amostras que compõe o sinal.
            samples_per_cycle: a quantidade de amostras de sinal por ciclo do sinal.
        '''
        self.signal = signal.reshape(-1,)
        self.fourier_filter = FourierFilter(samples_per_cycle)
        self.num_points = num_points

    def estimate(self):
        '''
        Estima um fasor utilizando a convolução do sinal com os filtros de Fourier cosseno e seno.
        '''
        self.real_part = np.convolve(self.signal, self.fourier_filter.cosine_filter)
        self.imaginary_part = np.convolve(self.signal, self.fourier_filter.sine_filter)
        self.amplitude = np.sqrt(self.real_part**2 + self.imaginary_part**2)[:self.num_points]
        phase_rad = np.arctan2(self.imaginary_part, self.real_part)[:self.num_points]
        self.phase = np.degrees(phase_rad)
        self.complex = self.amplitude * np.exp(1j * phase_rad)


class Iec:
    curves_params = {
        'IEC_normal_inverse': {'k': 0.14, 'a': 0.02, 'c': 0},
        'IEC_very_inverse': {'k': 13.5, 'a': 1, 'c': 0},
        'IEC_extremely_inverse': {'k': 80, 'a': 2, 'c': 0},
        'UK_long_time_inverse': {'k': 120, 'a': 1, 'c': 0},
        'IEEE_moderately_inverse': {'k': 0.0515, 'a': 0.02, 'c': 0.114},
        'IEEE_very_inverse': {'k': 19.61, 'a': 2, 'c': 0.491},
        'IEEE_extremely_inverse': {'k': 28.2, 'a': 2, 'c': 0.1217},
        'US_CO8_inverse': {'k': 5.95, 'a': 2, 'c': 0.18},
        'US_CO2_short_time_inverse': {'k': 0.02394, 'a': 0.02, 'c': 0.01694},
    }

    def __init__(self, va, vb, vc, ia, ib, ic, t, sampling_period, b, c, md,
                 R, XL, estimator_samples_per_cycle, RTC, frequency=60):
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
        self.estimator_samples_per_cycle = estimator_samples_per_cycle
        self.RTC = RTC
        self.frequency = frequency
        self._apply_anti_aliasing_filter()
        self._resample()
        self._apply_mimic_filter()
        self._estimate_phasors()

    def _apply_anti_aliasing_filter(self):
        for signal in self.signals:
            anti_aliasing_filter = AntiAliasingFilter(self.sampling_period, self.signals[signal], self.b, self.c)
            anti_aliasing_filter.apply_filter()
            self.signals[signal] = anti_aliasing_filter.filtered_signal

    def _resample(self):
        for signal in self.signals:
            self.signals[signal] = self.signals[signal][::self.md].reshape(-1)
        self.time = np.array(self.time[::self.md]).reshape(-1)
        self.sampling_period = self.time[1] - self.time[0]

    def _apply_mimic_filter(self):
        inductance = self.XL / (2 * np.pi * self.frequency)
        tau = (inductance / self.R) / self.sampling_period
        for signal in self.signals:
            mimic_filter = MimicFilter(self.signals[signal], tau, self.sampling_period)
            mimic_filter.apply_filter()
            self.signals[signal] = mimic_filter.filtered_signal

    def _estimate_phasors(self):
        self.phasors = {
            'va': PhasorEstimator(self.signals['va'], self.estimator_samples_per_cycle, len(self.time)),
            'vb': PhasorEstimator(self.signals['vb'], self.estimator_samples_per_cycle, len(self.time)),
            'vc': PhasorEstimator(self.signals['vc'], self.estimator_samples_per_cycle, len(self.time)),
            'v0': PhasorEstimator(np.zeros_like(self.signals['va']), self.estimator_samples_per_cycle, len(self.time)),
            'ia': PhasorEstimator(self.signals['ia'], self.estimator_samples_per_cycle, len(self.time)),
            'ib': PhasorEstimator(self.signals['ib'], self.estimator_samples_per_cycle, len(self.time)),
            'ic': PhasorEstimator(self.signals['ic'], self.estimator_samples_per_cycle, len(self.time)),
            'i0': PhasorEstimator(np.zeros_like(self.signals['ia']), self.estimator_samples_per_cycle, len(self.time)),
        }
        for signal in self.phasors:
            if signal == 'v0':
                self.phasors['v0'].amplitude = abs(
                    self.phasors['va'].complex + self.phasors['vb'].complex + self.phasors['vc'].complex
                )
                self.phasors['v0'].phase = np.degrees(
                    np.angle(self.phasors['va'].complex + self.phasors['vb'].complex + self.phasors['vc'].complex)
                )
            elif signal == 'i0':
                self.phasors['i0'].amplitude = abs(
                    self.phasors['ia'].complex + self.phasors['ib'].complex + self.phasors['ic'].complex
                )
                self.phasors['i0'].phase = np.degrees(
                    np.angle(self.phasors['ia'].complex + self.phasors['ib'].complex + self.phasors['ic'].complex)
                )
            else:
                self.phasors[signal].estimate()

    def add_51(self, type, gamma, adjust_current, curve):
        if type == 'phase':
            k = self.curves_params[curve]['k']
            a = self.curves_params[curve]['a']
            c = self.curves_params[curve]['c']
            trip_times = {'ia': np.array([]), 'ib': np.array([]), 'ic': np.array([])}

            for current in ['ia', 'ib', 'ic']:
                i_t = (self.phasors[current].amplitude, self.time)
                i_t[0][i_t[0] == adjust_current] = adjust_current + 0.01  # Avoid division by zero
                curve_common_term = gamma * (k / (((i_t[0] / self.RTC) / adjust_current) ** a - 1) + c)
                trip_times[current] = np.where((i_t[0] / self.RTC) <= adjust_current, np.inf, i_t[1] + curve_common_term)
            self.min_time_51F = {current: np.min(trip_times[current]) for current in ['ia', 'ib', 'ic']}

        elif type == 'neutral':
            k = self.curves_params[curve]['k']
            a = self.curves_params[curve]['a']
            c = self.curves_params[curve]['c']
            trip_times = np.array([])

            i_t = (self.phasors['i0'].amplitude, self.time)
            i_t[0][i_t[0] == adjust_current] = adjust_current + 0.01
            curve_common_term = gamma * (k / (((i_t[0] / self.RTC) / adjust_current) ** a - 1) + c)
            trip_times = np.where((i_t[0] / self.RTC) <= adjust_current, np.inf, i_t[1] + curve_common_term)
            self.min_time_51N = np.min(trip_times)

    def add_50(self, type, adjust_current):
        if type == 'phase':
            relays_trip_times = {'a': np.array([]),
                                 'b': np.array([]),
                                 'c': np.array([]),
                                 'b\'': np.array([]),
                                 'c\'': np.array([]),
                                 'd\'': np.array([])}

            trip_times = {'ia': relays_trip_times.copy(),
                          'ib': relays_trip_times.copy(),
                          'ic': relays_trip_times.copy()}

            self.min_time_50F = {'ia': np.inf, 'ib': np.inf, 'ic': np.inf}

            for current in ['ia', 'ib', 'ic']:
                for relay in ['a', 'b', 'c', 'b\'', 'c\'', 'd\'']:
                    i_t = (self.phasors[current].amplitude, self.time)
                    i_t[0][i_t[0] == adjust_current[relay]] = adjust_current[relay] + 0.01
                    trip_times[current][relay] = np.where(i_t[0] <= adjust_current[relay], np.inf, i_t[1])
                    curr_relay_min_time = np.min(trip_times[current][relay])
                    if curr_relay_min_time < self.min_time_50F[current]:
                        self.min_time_50F[current] = curr_relay_min_time

        elif type == 'neutral':
            trip_times = {'a': np.array([]),
                          'b': np.array([]),
                          'c': np.array([]),
                          'b\'': np.array([]),
                          'c\'': np.array([]),
                          'd\'': np.array([])}
            for relay in ['a', 'b', 'c', 'b\'', 'c\'', 'd\'']:
                i_t = (self.phasors['i0'].amplitude, self.time)
                i_t[0][i_t[0] == adjust_current[relay]] = adjust_current[relay] + 0.01
                trip_times[relay] = np.where(i_t[0] <= adjust_current[relay], np.inf, i_t[1])
            self.min_time_50N = min([np.min(trip_times[relay]) for relay in ['a', 'b', 'c', 'b\'', 'c\'', 'd\'']])

    def add_32(self, type, alpha, beta):
        if type == 'phase':
            if alpha == 30:
                pass
            elif alpha == 60:
                pass
            elif alpha == 90:
                v_pol_phases = {
                    'a': np.degrees(np.angle(self.phasors['vb'].complex - self.phasors['vc'].complex)),
                    'b': np.degrees(np.angle(self.phasors['vc'].complex - self.phasors['va'].complex)),
                    'c': np.degrees(np.angle(self.phasors['va'].complex - self.phasors['vb'].complex)),
                }

                i_op_phases = {
                    'a': self._replace_after_transitions(self.phasors['ia'].phase, 0),
                    'b': self._replace_after_transitions(self.phasors['ib'].phase, 0),
                    'c': self._replace_after_transitions(self.phasors['ic'].phase, 0),
                }

                op_region = {
                    'a': (v_pol_phases['a'] - 90 + beta, v_pol_phases['a'] + 90 + beta),
                    'b': (v_pol_phases['b'] - 90 + beta, v_pol_phases['b'] + 90 + beta),
                    'c': (v_pol_phases['c'] - 90 + beta, v_pol_phases['c'] + 90 + beta),
                }
                self.trip_permission_32F = {
                    'a': ((i_op_phases['a'] > op_region['a'][0]) | np.isnan(i_op_phases['a'])) &
                         ((i_op_phases['a'] < op_region['a'][1]) | np.isnan(i_op_phases['a'])),
                    'b': ((i_op_phases['b'] > op_region['b'][0]) | np.isnan(i_op_phases['b'])) &
                         ((i_op_phases['b'] < op_region['b'][1]) | np.isnan(i_op_phases['b'])),
                    'c': ((i_op_phases['c'] > op_region['c'][0]) | np.isnan(i_op_phases['c'])) &
                         ((i_op_phases['c'] < op_region['c'][1]) | np.isnan(i_op_phases['c'])),
                }

        elif type == 'neutral':
            if alpha == 30:
                pass
            elif alpha == 60:
                pass
            elif alpha == 90:
                v_pol_phases = -self.phasors['v0'].phase
                i_op_phases = self._replace_after_transitions(self.phasors['i0'].phase, 0)
                op_region = (-v_pol_phases - 90 + beta, -v_pol_phases + 90 + beta)
                self.trip_permission_32N = ((i_op_phases > op_region[0]) | np.isnan(i_op_phases)) & \
                                           ((i_op_phases < op_region[1]) | np.isnan(i_op_phases))

    def add_67(self, type, gamma, timed_adjust_current, insta_adjust_current, curve):
        if type == 'phase':
            self.add_51(type='phase', gamma=gamma, adjust_current=timed_adjust_current, curve=curve)
            self.add_50(type='phase', adjust_current=insta_adjust_current)
            self.add_32(type='phase', alpha=90, beta=30)
            self.logical_state_51F = {
                'ia': [0 if t < self.min_time_51F['ia'] else 1 for t in self.time],
                'ib': [0 if t < self.min_time_51F['ib'] else 1 for t in self.time],
                'ic': [0 if t < self.min_time_51F['ic'] else 1 for t in self.time],
            }
            self.logical_state_50F = {
                'ia': [0 if t < self.min_time_50F['ia'] else 1 for t in self.time],
                'ib': [0 if t < self.min_time_50F['ib'] else 1 for t in self.time],
                'ic': [0 if t < self.min_time_50F['ic'] else 1 for t in self.time],
            }
            self.trip_permission_67F = {
                'a': ((self.logical_state_51F['ia'] & self.trip_permission_32F['a']) |
                      (self.logical_state_50F['ia'] & self.trip_permission_32F['a'])),
                'b': ((self.logical_state_51F['ib'] & self.trip_permission_32F['b']) |
                      (self.logical_state_50F['ib'] & self.trip_permission_32F['b'])),
                'c': ((self.logical_state_51F['ic'] & self.trip_permission_32F['c']) |
                      (self.logical_state_50F['ic'] & self.trip_permission_32F['c'])),
            }

        elif type == 'neutral':
            self.add_51(type='neutral', gamma=gamma, adjust_current=timed_adjust_current, curve=curve)
            self.add_50(type='neutral', adjust_current=insta_adjust_current)
            self.add_32(type='neutral', alpha=90, beta=30)
            self.logical_state_51N = [0 if t < self.min_time_51N else 1 for t in self.time]
            self.logical_state_50N = [0 if t < self.min_time_50N else 1 for t in self.time]
            self.trip_permission_67N = ((self.logical_state_51N & self.trip_permission_32N) |
                                        (self.logical_state_50N & self.trip_permission_32N))

    @property
    def trip_signal(self):
        return (self.trip_permission_67F['a'] | self.trip_permission_67F['b'] |
                self.trip_permission_67F['c'] | self.trip_permission_67N)

    def _replace_after_transitions(self, arr, n):
        arr = np.array(arr, dtype=float)
        length = len(arr)

        for i in range(length - 1):
            if arr[i] > 0 and arr[i + 1] < 0:
                start_index = max(i - n + 1, 0)
                arr[start_index:i + 1] = np.nan

                end_index = min(i + 1 + n, length)
                arr[i + 1:end_index] = np.nan
        return arr
