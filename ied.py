import numpy as np
from typing import Sequence
from phasor_estimator import PhasorEstimator
from filters import AntiAliasingFilter, MimicFilter
from curves import Curve


class Ied:
    def __init__(self, va: Sequence[float], vb: Sequence[float], vc: Sequence[float],
                 ia: Sequence[float], ib: Sequence[float], ic: Sequence[float],
                 t: Sequence[float], sampling_period: float,
                 b: float, c: float, md: int,
                 R: float, XL: float,
                 estimator_sample_rate: int, RTC: int, frequency: int = 60):
        '''
        Instancia um objeto Ied.
        Args:
            va (Sequence[float]): Sinal de tensão fase A.
            vb (Sequence[float]): Sinal de tensão fase B.
            vc (Sequence[float]): Sinal de tensão fase C.
            ia (Sequence[float]): Sinal de corrente fase A.
            ib (Sequence[float]): Sinal de corrente fase B.
            ic (Sequence[float]): Sinal de corrente fase C.
            t (Sequence[float]): Vetor de tempo.
            sampling_period (float): Período de amostragem.
            b (float): Parâmetro do filtro de antialiasing.
            c (float): Parâmetro do filtro de antialiasing.
            md (int): Fator de downsampling.
            R (float): Resistência do circuito.
            XL (float): Reatância indutiva do circuito.
            estimator_sample_rate (int): Número de amostras por ciclo para o estimador de fasores.
            RTC (int): Relação de transformação de corrente.
            frequency (int): Frequência da rede elétrica.
        '''
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
        self.md = md
        self.R = R
        self.XL = XL
        self.estimator_sample_rate = estimator_sample_rate
        self.RTC = RTC
        self.frequency = frequency
        self._apply_anti_aliasing_filter()
        self._resample()
        self._apply_mimic_filter()
        self._estimate_phasors()

    def _apply_anti_aliasing_filter(self):
        '''
        Aplica um filtro de antialiasing aos sinais de tensão e corrente.
        '''
        for signal in self.signals:
            anti_aliasing_filter = AntiAliasingFilter(self.sampling_period, self.signals[signal], self.b, self.c)
            anti_aliasing_filter.apply_filter()
            self.signals[signal] = anti_aliasing_filter.filtered_signal

    def _resample(self):
        '''
        Realiza o downsampling dos sinais de tensão e corrente.
        '''
        for signal in self.signals:
            self.signals[signal] = self.signals[signal][::self.md].reshape(-1)
        self.time = np.array(self.time[::self.md]).reshape(-1)
        self.sampling_period = self.time[1] - self.time[0]

    def _apply_mimic_filter(self):
        '''
        Aplica o filtro mímico aos sinais de tensão e corrente.
        '''
        inductance = self.XL / (2 * np.pi * self.frequency)
        tau = (inductance / self.R) / self.sampling_period
        for signal in self.signals:
            mimic_filter = MimicFilter(self.signals[signal], tau, self.sampling_period)
            mimic_filter.apply_filter()
            self.signals[signal] = mimic_filter.filtered_signal

    def _estimate_phasors(self):
        '''
        Estima os fasores de tensão e corrente.
        '''
        self.phasors = {
            'va': PhasorEstimator(self.signals['va'], self.estimator_sample_rate, len(self.time)),
            'vb': PhasorEstimator(self.signals['vb'], self.estimator_sample_rate, len(self.time)),
            'vc': PhasorEstimator(self.signals['vc'], self.estimator_sample_rate, len(self.time)),
            'v0': PhasorEstimator(np.zeros_like(self.signals['va']), self.estimator_sample_rate, len(self.time)),
            'ia': PhasorEstimator(self.signals['ia'], self.estimator_sample_rate, len(self.time)),
            'ib': PhasorEstimator(self.signals['ib'], self.estimator_sample_rate, len(self.time)),
            'ic': PhasorEstimator(self.signals['ic'], self.estimator_sample_rate, len(self.time)),
            'i0': PhasorEstimator(np.zeros_like(self.signals['ia']), self.estimator_sample_rate, len(self.time)),
        }

        for signal in self.phasors:
            if signal not in ['v0', 'i0']:
                self.phasors[signal].estimate()

        self.phasors['v0'].exp_form = (self.phasors['va'].exp_form + self.phasors['vb'].exp_form + self.phasors['vc'].exp_form)
        self.phasors['v0'].amplitude = abs(self.phasors['va'].exp_form + self.phasors['vb'].exp_form + self.phasors['vc'].exp_form)
        self.phasors['v0'].phase = np.degrees(np.angle(self.phasors['va'].exp_form + self.phasors['vb'].exp_form
                                                       + self.phasors['vc'].exp_form))

        self.phasors['i0'].exp_form = (self.phasors['ia'].exp_form + self.phasors['ib'].exp_form + self.phasors['ic'].exp_form)
        self.phasors['i0'].amplitude = abs(self.phasors['ia'].exp_form + self.phasors['ib'].exp_form + self.phasors['ic'].exp_form)
        self.phasors['i0'].phase = np.degrees(np.angle(self.phasors['ia'].exp_form + self.phasors['ib'].exp_form +
                                                       self.phasors['ic'].exp_form))

    def add_51(self, type, gamma, adjust_current, curve):
        '''
        Adiciona um elemento de proteção 51 ao sistema.
        Args:
            type: O tipo do elemento de proteção (fase ou neutro).
            gamma: O multiplicador do tempo de ajuste do relé 51.
            adjust_current: A corrente de ajuste do relé 51.
            curve: A curva do relé 51.
        '''
        (k, a, c) = Curve[curve.upper()].value

        if type == 'phase':
            trip_times = {'ia': np.array([]), 'ib': np.array([]), 'ic': np.array([])}

            for current in ['ia', 'ib', 'ic']:
                i_t = (self.phasors[current].amplitude, self.time)
                i_t[0][i_t[0] == adjust_current] = adjust_current + 0.01  # Evita divisão por zero
                curve_common_term = gamma * (k / (((i_t[0] / self.RTC) / adjust_current) ** a - 1) + c)
                trip_times[current] = np.where((i_t[0] / self.RTC) <= adjust_current, np.inf, i_t[1] + curve_common_term)
            self.min_time_51F = {current: np.min(trip_times[current]) for current in ['ia', 'ib', 'ic']}

        elif type == 'neutral':
            trip_times = np.array([])

            i_t = (self.phasors['i0'].amplitude, self.time)
            i_t[0][i_t[0] == adjust_current] = adjust_current + 0.01
            curve_common_term = gamma * (k / (((i_t[0] / self.RTC) / adjust_current) ** a - 1) + c)
            trip_times = np.where((i_t[0] / self.RTC) <= adjust_current, np.inf, i_t[1] + curve_common_term)
            self.min_time_51N = np.min(trip_times)

    def add_50(self, type, adjust_current):
        '''
        Adiciona um elemento de proteção 50 ao sistema.
        Args:
            type: O tipo do elemento de proteção (fase ou neutro).
            adjust_current: A corrente de ajuste do relé 50.
        '''
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
                    i_t[0][i_t[0] == adjust_current[relay]] = adjust_current[relay] + 0.01  # Evita divisão por zero
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
        '''
        Adiciona um elemento de proteção 32 ao sistema.
        Args:
            type: O tipo do elemento de proteção (fase ou neutro).
            alpha: O ângulo de ajuste do relé 32.
            beta: O ângulo de máximo torque do relé 32.
        '''
        if type == 'phase':
            if alpha == 30:
                pass
            elif alpha == 60:
                pass
            elif alpha == 90:
                v_pol_phases = {
                    'a': np.degrees(np.angle(self.phasors['vb'].exp_form - self.phasors['vc'].exp_form)),
                    'b': np.degrees(np.angle(self.phasors['vc'].exp_form - self.phasors['va'].exp_form)),
                    'c': np.degrees(np.angle(self.phasors['va'].exp_form - self.phasors['vb'].exp_form)),
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
        '''
        Adiciona um elemento de proteção 67 ao sistema.
        Args:
            type: O tipo do elemento de proteção (fase ou neutro).
            gamma: O multiplicador do tempo de ajuste do relé 51.
            timed_adjust_current: A corrente de ajuste do relé 51.
            insta_adjust_current: A corrente de ajuste do relé 50.
            curve: A curva do relé 51.
        '''
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

    def add_21(self, inclination_angle, zones_impedances, line_z1, line_z0):
        self.distance_trip_signals = {
            'at': {'zone1': None, 'zone2': None, 'zone3': None},
            'bt': {'zone1': None, 'zone2': None, 'zone3': None},
            'ct': {'zone1': None, 'zone2': None, 'zone3': None},
            'ab': {'zone1': None, 'zone2': None, 'zone3': None},
            'bc': {'zone1': None, 'zone2': None, 'zone3': None},
            'ca': {'zone1': None, 'zone2': None, 'zone3': None},
        }
        self.measured_impedances = {'at': None, 'bt': None, 'ct': None, 'ab': {}, 'bc': {}, 'ca': {}}

        s_op = {'at': {}, 'bt': {}, 'ct': {}, 'ab': {}, 'bc': {}, 'ca': {}}

        impedance = {'zone1': zones_impedances[0], 'zone2': zones_impedances[1], 'zone3': zones_impedances[2]}
        k = (line_z0 - line_z1) / line_z1

        for unit in ['at', 'bt', 'ct', 'ab', 'bc', 'ca']:
            if unit in ['at', 'bt', 'ct']:
                vr = self.phasors['v' + unit[0]].exp_form[16:]
                ir = self.phasors['i' + unit[0]].exp_form[16:] + k * (1/3) * self.phasors['i0'].exp_form[16:]
                self.measured_impedances[unit] = vr / ir
                v_pol = 'bc' if unit == 'at' else 'ca' if unit == 'bt' else 'ab'
                s_pol = 1j * self.phasors['v' + v_pol[0]].exp_form[16:] - self.phasors['v' + v_pol[1]].exp_form[16:]

            if unit in ['ab', 'bc', 'ca']:
                ir = self.phasors['i' + unit[0]].exp_form[16:] - self.phasors['i' + unit[1]].exp_form[16:]
                vr = self.phasors['v' + unit[0]].exp_form[16:] - self.phasors['v' + unit[1]].exp_form[16:]
                self.measured_impedances[unit] = vr / ir
                v_pol = 'c' if unit == 'ab' else 'a' if unit == 'bc' else 'b'
                s_pol = -1j * self.phasors['v' + v_pol].exp_form[16:]

            for zone in ['zone1', 'zone2', 'zone3']:
                s_op[unit][zone] = (
                    np.abs(impedance[zone]) * (np.cos(inclination_angle) + 1j * np.sin(inclination_angle)) * ir - vr
                )

                cos_comparator = np.real(s_op[unit][zone] * np.conj(s_pol))
                normalized_cos_comparator = cos_comparator / (np.abs(cos_comparator) + 1e-15)
                self.distance_trip_signals[unit][zone] = np.where(normalized_cos_comparator >= 0.99, 1, 0)

    @property
    def trip_signal(self):
        '''
        Retorna o sinal de disparo do IED
        '''
        return (self.trip_permission_67F['a'] | self.trip_permission_67F['b'] |
                self.trip_permission_67F['c'] | self.trip_permission_67N)

    def _replace_after_transitions(self, arr, n):
        '''
        Substitui n amostras após uma transição + -> - por NaN.
        '''
        arr = np.array(arr, dtype=float)
        length = len(arr)

        for i in range(length - 1):
            if arr[i] > 0 and arr[i + 1] < 0:
                start_index = max(i - n + 1, 0)
                arr[start_index:i + 1] = np.nan

                end_index = min(i + 1 + n, length)
                arr[i + 1:end_index] = np.nan
        return arr
