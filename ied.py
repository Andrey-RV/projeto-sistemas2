from __future__ import annotations
import numpy as np
import numpy.typing as npt
import relays
from scipy.signal import fftconvolve  # type: ignore
from signals import Signals
from filters import AntiAliasingFilter, FourierFilter, MimicFilter

RelayType = (relays.Neutral50 | relays.Phase50 | relays.Neutral51 | relays.Phase51 | relays.Phase32 | relays.Neutral32 |
             relays.Phase67 | relays.Neutral67 | relays.Relay21 | None)


class PhasorEstimator:
    def __init__(self, samples_per_cycle: int) -> None:
        '''
        Instancia um estimador de fasor.

        Args:
            samples_per_cycle (int): A quantidade de amostras capturadas pelo IED em um período da onda fundamental.
        '''
        self._samples_per_cycle = samples_per_cycle
        self._fourier_filters = FourierFilter(self._samples_per_cycle)
        self._fourier_filters.create_filter()

    def estimate(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        '''
        Retorna a representação complexa do fasor estimado.

        Args:
            signal (npt.NDArray[np.float64]): O sinal base para a estimativa do fasor.

        Returns:
            npt.NDArray[np.complex128]: O fasor estimado.
        '''

        real = fftconvolve(signal, self._fourier_filters.cosine_filter)[:len(signal)]
        imaginary = fftconvolve(signal, self._fourier_filters.sine_filter)[:len(signal)]
        return real + 1j * imaginary


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
        RTC: float = 1,
        RTPC: float = 1,
        frequency: int = 60,
        should_be_referred: bool = False,
    ) -> None:
        '''Instancia um objeto Ied.
        Args:
            signals (Signals): Uma instância de Signals contendo os sinais de tensão, corrente, tempo e período de
            amostragem.
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
        self._relays: dict[str, RelayType] = {
            key: None for key in ['50N', '50F', '51N', '51F', '32F', '32N', '67F', '67N', '21']
        }

        self._trips = {  # type: ignore
            key: {} for key in ['50N', '50F', '51N', '51F', '32F', '32N', '67F', '67N', '21']
        }

        self._refer_to_secondary(should_be_referred)
        self._apply_anti_aliasing_filter()
        self._resample_signals()
        self._apply_mimic_filter()
        self._estimate_phasors()

    @property
    def phasors(self) -> Signals:
        return self._phasors

    @property
    def relays(self) -> dict[str, RelayType]:
        return self._relays

    def _refer_to_secondary(self, should_be_referred: bool) -> None:
        '''Atualiza os sinais para a referência secundária caso os dados passados para o IED estejam na referência
        primária.
        Args:
            should_be_referred (bool): Indica se os sinais devem ser referenciados para a secundária.

        Returns:
            None
        '''
        if should_be_referred:
            for current_name, current_value in self._signals.get_currents():
                self._signals[current_name] = current_value / self._RTC  # type: ignore

            for voltage_name, voltage_value in self._signals.get_voltages():
                self._signals[voltage_name] = voltage_value / self._RTPC  # type: ignore

    def _apply_anti_aliasing_filter(self) -> None:
        '''Aplica o filtro de antialiasing aos sinais de tensão e corrente com o auxílio da classe AntiAliasingFilter,
        salvando os sinais filtrados na propriedade aa_signals.

        Returns:
            None
        '''
        self._aa_filter = AntiAliasingFilter(self._signals.sampling_period, self._b, self._c)
        self._aa_signals = Signals(t=self._signals.t, sampling_period=self._signals.sampling_period)

        for signal_name, signal_data in self._signals:
            self._aa_signals[signal_name] = self._aa_filter.apply_filter(signal_data)  # type: ignore

    def _resample_signals(self) -> None:
        '''Realiza o downsampling dos sinais aa_signals com o fator de decimação md e salva os novos sinais na
        propriedade resampled_aa_signals.

        Returns:
            None
        '''
        new_t = self._aa_signals.t[:: self._md]
        new_sampling_period = self._aa_signals.sampling_period * self._md
        self._resampled_aa_signals = Signals(t=new_t, sampling_period=new_sampling_period)

        for name, data in self._aa_signals:
            self._resampled_aa_signals[name] = np.array(data[:: self._md])

    def _apply_mimic_filter(self) -> None:
        '''Aplica o filtro MIMIC aos sinais resampledados e salva os sinais filtrados na propriedade
        mimic_filtered_signals.

        Returns:
            None
        '''
        inductance = self._XL / (2 * np.pi * self._frequency)
        tau = (inductance / self._R) / self._resampled_aa_signals.sampling_period
        self._mimic_filter = MimicFilter(tau, self._resampled_aa_signals.sampling_period)

        self._mimic_filtered_signals = Signals(
            t=self._resampled_aa_signals.t,
            sampling_period=self._resampled_aa_signals.sampling_period
        )

        for signal_name, signal_data in self._resampled_aa_signals:
            self._mimic_filtered_signals[signal_name] = self._mimic_filter.apply_filter(signal_data)  # type: ignore

    def _estimate_phasors(self) -> None:
        '''Estima os fasores de tensão e corrente com o auxílio da classe PhasorEstimator.

        Returns:
            None
        '''
        estimator = PhasorEstimator(self._samples_per_cycle)
        self._phasors = Signals(
            t=self._mimic_filtered_signals.t,
            sampling_period=self._mimic_filtered_signals.sampling_period,
        )

        for name, data in self._mimic_filtered_signals:
            if name not in ['vn', 'in']:
                self._phasors[name] = estimator.estimate(data)  # type: ignore

        self._phasors['vn'] = (self._phasors.va + self._phasors.vb + self._phasors.vc)
        self._phasors['in'] = (self._phasors.ia + self._phasors.ib + self._phasors.ic)

    def add_relay(self, relay_type: str, **kwargs) -> None:
        """Adiciona um relé ao IED.

        Args:
            relay_type (str): O tipo de relé a ser adicionado. Pode ser '50N', '50F', '51N', '51F', '32F', '32N', '67F',
            '67N' ou '21'.
            **kwargs: Argumentos adicionais a serem passados para os construtores dos relés.

        Raises:
            NotImplementedError: Caso o tipo de relé passado não esteja implementado.
        """
        relay_instance: RelayType
        match relay_type:
            case '51N':
                relay_instance = relays.Neutral51(
                    time_vector=self.phasors.t,
                    neutral_current=self._phasors['in'].astype(np.complex128),  # type: ignore
                    **kwargs,
                )
            case '51F':
                relay_instance = relays.Phase51(
                    time_vector=self.phasors.t,
                    current_phasors=self._phasors,
                    **kwargs,
                )
            case '50N':
                relay_instance = relays.Neutral50(
                    time_vector=self.phasors.t,
                    neutral_current=self._phasors['in'].astype(np.complex128),  # type: ignore
                    **kwargs,
                )
            case '50F':
                relay_instance = relays.Phase50(
                    time_vector=self.phasors.t,
                    current_phasors=self._phasors,
                    **kwargs,
                )
            case '32F':
                relay_instance = relays.Phase32(
                    phasors=self._phasors,
                    **kwargs,
                )
            case '32N':
                relay_instance = relays.Neutral32(
                    zero_seq_current=(1 / 3) * self._phasors['in'],
                    zero_seq_voltage=(1 / 3) * self._phasors['vn'],
                    **kwargs,
                )
            case '67F':
                relay_instance = relays.Phase67(
                    trips=self._trips,
                    time_vector=self.phasors.t,
                )
            case '67N':
                relay_instance = relays.Neutral67(
                    trips=self._trips,
                    time_vector=self.phasors.t,
                )
            case '21':
                relay_instance = relays.Relay21(
                    phasors=self._phasors,
                    **kwargs,
                )
            case _:
                raise NotImplementedError(f"Relay type {relay_type} not implemented.")

        self._relays[relay_type] = relay_instance
        self._trips[relay_type] = relay_instance.analyze_trip()

    @property
    def trip_signal(self) -> npt.NDArray[np.bool_]:
        return (
            self._trips['67N']['neutral'] |
            self._trips['67F']['ia'] |
            self._trips['67F']['ib'] |
            self._trips['67F']['ic']
        )
