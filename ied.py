import numpy as np
import relays
from signals import Signals
from phasor_estimator import PhasorEstimator
from filters import AntiAliasingFilter, MimicFilter


class Ied:
    def __init__(
        self,
        signals: Signals,
        b: float,
        c: float,
        md: int,
        R: float,
        XL: float,
        samples_per_cycle: int,
        RTC: int,
        RTPC: int = 0,
        frequency: int = 60,
        should_be_referred: bool = False,
    ) -> None:
        '''Instancia um objeto Ied.
        Args:
            signals (Signals): Uma instância de Signals contendo os sinais de tensão, corrente, tempo e período de amostragem.
            b (float): Parâmetro do filtro de antialiasing.
            c (float): Parâmetro do filtro de antialiasing.
            md (int): Fator de downsampling.
            R (float): Resistência do circuito.
            XL (float): Reatância indutiva do circuito.
            samples_per_cycle (int): A quantidade de amostras capturadas pelo IED em um período da onda fundamental.
            RTC (int): Relação de transformação de corrente.
            RTPC (int): Relação de transformação de potencial capacitivo.
            frequency (int): Frequência da rede elétrica.
            should_be_referred (bool): Indica se os sinais devem ser referenciados para o secundário. Default é False.

        Returns:
            None
        '''
        self._signals = signals
        self._b = b
        self._c = c
        self._md = md
        self._R = R
        self._XL = XL
        self._samples_per_cycle = samples_per_cycle
        self._RTC = RTC
        self._RTPC = RTPC
        self._frequency = frequency
        self.__relays: dict = {
            '50N': {},
            '50F': {},
            '51N': {},
            '51F': {},
            '32F': {},
            '32N': {},
            '67': {},
            '21': {},
        }
        self._trips: dict = {
            '50N': {},
            '50F': {},
            '51N': {},
            '51F': {},
            '32F': {},
            '32N': {},
            '67': {},
            '21': {},
        }
        self.__refer_to_secondary(should_be_referred)
        self.__apply_anti_aliasing_filter()
        self.__resample_signals()
        self.__apply_mimic_filter()
        self.__estimate_phasors()

    @property
    def phasors(self):
        return self.__phasors

    @property
    def relays(self):
        return self.__relays

    def __refer_to_secondary(self, should_be_referred: bool) -> None:
        '''Atualiza os sinais para a referência secundária caso os dados passados para o IED estejam na referência primária.
        Args:
            should_be_referred (bool): Indica se os sinais devem ser referenciados para a secundária.

        Returns:
            None
        '''
        if should_be_referred:
            for current_name, current_value in self._signals.get_currents():
                self._signals[current_name] = current_value / self._RTC

            for voltage_name, voltage_value in self._signals.get_voltages():
                self._signals[voltage_name] = voltage_value / self._RTPC

    def __apply_anti_aliasing_filter(self) -> None:
        '''Aplica o filtro de antialiasing aos sinais de tensão e corrente com o auxílio da classe AntiAliasingFilter, salvando os sinais filtrados na propriedade aa_signals.

        Returns:
            None
        '''
        self._anti_aliasing_filter = AntiAliasingFilter(self._signals.sampling_period, self._b, self._c)
        self._aa_signals = Signals(t=self._signals.t, sampling_period=self._signals.sampling_period)

        for signal_name, signal_data in self._signals:
            self._aa_signals[signal_name] = self._anti_aliasing_filter.apply_filter(signal_data)  # type: ignore

    def __resample_signals(self) -> None:
        '''Realiza o downsampling dos sinais aa_signals com o fator de decimação md e salva os novos sinais na propriedade resampled_aa_signals.

        Returns:
            None
        '''
        self._resampled_aa_signals = Signals(
                                        t=self._aa_signals.t[::self._md],
                                        sampling_period=self._aa_signals.sampling_period * self._md
                                    )

        for signal_name, signal_data in self._aa_signals:
            self._resampled_aa_signals[signal_name] = np.array(signal_data[::self._md])

    def __apply_mimic_filter(self) -> None:
        '''Aplica o filtro MIMIC aos sinais resampledados e salva os sinais filtrados na propriedade mimic_filtered_signals.

        Returns:
            None
        '''
        inductance = self._XL / (2 * np.pi * self._frequency)
        tau = (inductance / self._R) / self._resampled_aa_signals.sampling_period
        self._mimic_filter = MimicFilter(tau, self._resampled_aa_signals.sampling_period)
        self._mimic_filtered_signals = Signals(t=self._resampled_aa_signals.t, sampling_period=self._resampled_aa_signals.sampling_period)

        for signal_name, signal_data in self._resampled_aa_signals:
            self._mimic_filtered_signals[signal_name] = self._mimic_filter.apply_filter(signal_data)  # type: ignore

    def __estimate_phasors(self) -> None:
        '''Estima os fasores de tensão e corrente com o auxílio da classe PhasorEstimator.

        Returns:
            None
        '''
        self._phasor_estimator = PhasorEstimator(self._samples_per_cycle)
        self.__phasors = Signals(t=self._mimic_filtered_signals.t, sampling_period=self._mimic_filtered_signals.sampling_period)

        for signal_name, signal_data in self._mimic_filtered_signals:
            if signal_name not in ['v0', 'i0']:
                self.__phasors[signal_name] = self._phasor_estimator.estimate(signal_data)  # type: ignore

        self.__phasors['vn'] = (self.__phasors.va + self.__phasors.vb + self.__phasors.vc)
        self.__phasors['in'] = (self.__phasors.ia + self.__phasors.ib + self.__phasors.ic)

    def add_relay(self, relay_type: str, **kwargs):
        match relay_type:
            case '51N':
                self.__relays['51N'] = relays.NeutralRelay51(
                    time_vector=self._mimic_filtered_signals.t,
                    neutral_current=self.__phasors['in'].astype(np.complex128),  # Cast to avoid mypy error
                    **kwargs
                    )
            case '51F':
                self.__relays['51F'] = relays.PhaseRelay51(
                    time_vector=self._mimic_filtered_signals.t,
                    current_phasors=self.__phasors,
                    **kwargs
                    )
            case '50N':
                self.__relays['50N'] = relays.NeutralRelay50(
                    time_vector=self._mimic_filtered_signals.t,
                    neutral_current=self.__phasors['in'].astype(np.complex128),  # Cast to avoid mypy error
                    **kwargs
                    )
            case '50F':
                self.__relays['50F'] = relays.PhaseRelay50(
                    time_vector=self._mimic_filtered_signals.t,
                    current_phasors=self.__phasors,
                    **kwargs
                    )
            case '32F':
                self.__relays['32F'] = relays.PhaseRelay32(
                    phasors=self.__phasors,
                    **kwargs
                    )
            case '32N':
                self.__relays['32N'] = relays.NeutralRelay32(
                    zero_seq_current=1/3 * getattr(self.__phasors, 'in'),
                    zero_seq_voltage=1/3 * getattr(self.__phasors, 'vn'),
                    **kwargs
                    )
            case '67F':
                self.__relays['67F'] = relays.PhaseRelay67(
                    trips=self._trips,
                    time_vector=self._mimic_filtered_signals.t,
                    )
            case '67N':
                self.__relays['67N'] = relays.NeutralRelay67(
                    trips=self._trips,
                    time_vector=self._mimic_filtered_signals.t,
                    )
            case _:
                raise NotImplementedError(f"Relay type {relay_type} not implemented.")

        self._trips[relay_type] = self.__relays[relay_type].analyze_trip()

    # def add_21(self, inclination_angle, zones_impedances, line_z1, line_z0):
    #     self.distance_trip_signals = {
    #         'at': {'zone1': None, 'zone2': None, 'zone3': None},
    #         'bt': {'zone1': None, 'zone2': None, 'zone3': None},
    #         'ct': {'zone1': None, 'zone2': None, 'zone3': None},
    #         'ab': {'zone1': None, 'zone2': None, 'zone3': None},
    #         'bc': {'zone1': None, 'zone2': None, 'zone3': None},
    #         'ca': {'zone1': None, 'zone2': None, 'zone3': None},
    #     }
    #     self.measured_impedances = {'at': None, 'bt': None, 'ct': None, 'ab': {}, 'bc': {}, 'ca': {}}

    #     s_op = {'at': {}, 'bt': {}, 'ct': {}, 'ab': {}, 'bc': {}, 'ca': {}}

    #     impedance = {'zone1': zones_impedances[0], 'zone2': zones_impedances[1], 'zone3': zones_impedances[2]}
    #     k = (line_z0 - line_z1) / line_z1

    #     for unit in ['at', 'bt', 'ct', 'ab', 'bc', 'ca']:
    #         if unit in ['at', 'bt', 'ct']:
    #             vr = self.phasors['v' + unit[0]].exp_form[16:]
    #             ir = self.phasors['i' + unit[0]].exp_form[16:] + k * (1/3) * self.phasors['i0'].exp_form[16:]
    #             self.measured_impedances[unit] = vr / ir
    #             v_pol = 'bc' if unit == 'at' else 'ca' if unit == 'bt' else 'ab'
    #             s_pol = 1j * self.phasors['v' + v_pol[0]].exp_form[16:] - self.phasors['v' + v_pol[1]].exp_form[16:]

    #         if unit in ['ab', 'bc', 'ca']:
    #             ir = self.phasors['i' + unit[0]].exp_form[16:] - self.phasors['i' + unit[1]].exp_form[16:]
    #             vr = self.phasors['v' + unit[0]].exp_form[16:] - self.phasors['v' + unit[1]].exp_form[16:]
    #             self.measured_impedances[unit] = vr / ir
    #             v_pol = 'c' if unit == 'ab' else 'a' if unit == 'bc' else 'b'
    #             s_pol = -1j * self.phasors['v' + v_pol].exp_form[16:]

    #         for zone in ['zone1', 'zone2', 'zone3']:
    #             s_op[unit][zone] = (
    #                 np.abs(impedance[zone]) * (np.cos(inclination_angle) + 1j * np.sin(inclination_angle)) * ir - vr
    #             )

    #             cos_comparator = np.real(s_op[unit][zone] * np.conj(s_pol))
    #             normalized_cos_comparator = cos_comparator / (np.abs(cos_comparator) + 1e-15)
    #             self.distance_trip_signals[unit][zone] = np.where(normalized_cos_comparator >= 0.99, 1, 0)

    @property
    def trip_signal(self):
        '''
        Retorna o sinal de disparo do IED
        '''
        return (self._trips['67F']['ia'] | self._trips['67F']['ib'] | self._trips['67F']['ic']) | self._trips['67N']['neutral']
