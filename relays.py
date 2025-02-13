from __future__ import annotations
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, TypeAlias, Any
from signals import Signals


START_POINT: slice = slice(16, None)

number: TypeAlias = Union[Any, np.int64, np.float64, np.complex128]
numeric_array: TypeAlias = npt.NDArray[number]
StrNumberArrayDict: TypeAlias = dict[str, numeric_array]
StrBooleanDict: TypeAlias = dict[str, npt.NDArray[np.bool_]]
StrComplexDictOrNestedBoolDict: TypeAlias = Union[StrNumberArrayDict, dict[str, StrBooleanDict]]


class Curve(Enum):
    """Parâmetros das curvas de tempo dos relés 51 de tempo inverso (k, a, c) """
    IEC_NORMAL_INVERSE = (0.14, 0.02, 0)
    IEC_VERY_INVERSE = (13.5, 1, 0)
    IEC_EXTREMELY_INVERSE = (80, 2, 0)
    UK_LONG_TIME_INVERSE = (120, 1, 0)
    IEEE_MODERATELY_INVERSE = (0.0515, 0.02, 0.114)
    IEEE_VERY_INVERSE = (19.61, 2, 0.491)
    IEEE_EXTREMELY_INVERSE = (28.2, 2, 0.1217)
    US_CO8_INVERSE = (5.95, 2, 0.18)
    US_CO2_SHORT_TIME_INVERSE = (0.02394, 0.02, 0.01694)


class Relay21:
    def __init__(
        self,
        phasors: Signals,
        inclination_angle: float,
        line_positive_seq_impedance: complex,
        line_zero_seq_impedance: complex,
        abs_zones_impedances: dict[str, float]
    ) -> None:
        """Implementa a lógica de atuação do relé 21.

        Args:
            phasors (Signals): O conjunto trifásico de sinais de tensão e corrente.
            inclination_angle (float): O ângulo de inclinação do relé.
            line_positive_seq_impedance (complex): A impedância de sequência positiva da linha.
            line_zero_seq_impedance (complex): A impedância de sequência zero da linha.
            abs_zones_impedances (dict[str, float]): Os módulos das impedâncias das zonas de atuação do relé.
        """
        self._phasors = phasors
        self._inclination_angle = inclination_angle
        self._line_positive_seq_impedance = line_positive_seq_impedance
        self._line_zero_seq_impedance = line_zero_seq_impedance
        self._abs_zones_impedances = abs_zones_impedances

    def analyze_trip(self) -> dict[str, StrComplexDictOrNestedBoolDict]:
        """Retorna as impedâncias medidas e os sinais de trip.

        Returns:
            dict[str, StrComplexDictOrNestedBoolDict]: Um dicionário com as impedâncias medidas e os sinais de trip.
        """
        measured_impedances: dict[str, npt.NDArray[number]] = {}
        trip_signals: dict[str, StrBooleanDict] = {}

        k = ((self._line_zero_seq_impedance - self._line_positive_seq_impedance)
             / self._line_positive_seq_impedance)

        for unit in ["at", "bt", "ct", "ab", "bc", "ca"]:
            voltage, current = self.calculate_voltage_current(unit, k)
            measured_impedances[unit] = voltage / current
            trip_signals[unit] = self.analyze_via_comparator(unit, voltage, current)

        return {"measured_impedances": measured_impedances, "trip_signals": trip_signals}

    def calculate_voltage_current(self, unit: str, k: complex) -> tuple[numeric_array, numeric_array]:
        """Calcula a tensão e a corrente a ser usada no cálculo da unidade de impedância.

        Args:
            unit (str): A unidade de impedância a ser calculada.
            k (complex): O fator de correção da corrente de sequência zero.

        Returns:
            tuple[numeric_array, numeric_array]: A tensão e a corrente calculadas.
        """
        voltage: npt.NDArray[number]

        if "t" in unit:
            voltage = self._phasors[f"v{unit[0]}"][START_POINT]
            current = self._phasors[f"i{unit[0]}"][START_POINT] + k * (1 / 3) * self._phasors["in"][START_POINT]
        else:
            voltage = self._phasors[f"v{unit[0]}"][START_POINT] - self._phasors[f"v{unit[1]}"][START_POINT]
            current = self._phasors[f"i{unit[0]}"][START_POINT] - self._phasors[f"i{unit[1]}"][START_POINT]

        return voltage, current

    def analyze_via_comparator(self, unit: str, voltage: numeric_array, current: numeric_array) -> StrBooleanDict:
        """Analisa o sinal de trip de acordo com o comparador cosseno.

        Args:
            unit (str): A unidade de impedância a ser calculada.
            voltage (numeric_array): A tensão calculada.
            current (numeric_array): A corrente calculada.

        Returns:
            StrBooleanDict: O sinal de trip.
        """
        polarization_signal: npt.NDArray[Any]
        trip_signal: dict[str, npt.NDArray[np.bool_]] = {}

        if "t" in unit:
            phase = "bc" if unit == "at" else "ca" if unit == "bt" else "ab"
            polarization_signal = (
                1j * self._phasors[f"v{phase[0]}"][START_POINT]
                - self._phasors[f"v{phase[1]}"][START_POINT]
            )
        else:
            phase = "c" if unit == "ab" else "a" if unit == "bc" else "b"
            polarization_signal = -1j * self._phasors[f"v{phase}"][START_POINT]

        for zone, zone_impedance in self._abs_zones_impedances.items():
            op_signal = (
                zone_impedance
                * (np.cos(self._inclination_angle) + 1j * np.sin(self._inclination_angle))
                * current
                - voltage
            )
            cosine_comparator = np.real(op_signal * np.conj(polarization_signal))
            normalized = cosine_comparator / (np.abs(cosine_comparator) + 1e-6)
            trip_signal[zone] = normalized > 0.99

        return trip_signal


class _Relay32Base(ABC):
    def __init__(self, alpha: float, beta: float) -> None:
        """Classe abstrata para relés 32.

        Args:
            alpha (float): O ângulo alfa do relé.
            beta (float): O ângulo beta do relé.
        """
        self._alpha = alpha
        self._beta = beta

    @abstractmethod
    def analyze_trip(self) -> StrBooleanDict:
        """Método abstrato para análise de trip do relé 32.

        Raises:
            NotImplementedError: Deve ser implementado nas subclasses.

        Returns:
            StrBooleanDict: O sinal de trip.
        """
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")


class PhaseRelay32(_Relay32Base):
    def __init__(self, alpha: float, beta: float, phasors: Signals, ) -> None:
        """Implementa a lógica de atuação do relé 32 de fase.

        Args:
            alpha (float): O ângulo alfa do relé.
            beta (float): O ângulo beta do relé.
            phasors (Signals): O conjunto trifásico de sinais de tensão e corrente.
        """
        super().__init__(alpha, beta)
        self._phasors = phasors

    def analyze_trip(self) -> StrBooleanDict:
        """Analisa o sinal de trip do relé 32 de fase.

        Raises:
            NotImplementedError: Ângulo alfa igual a 30 ou 60 graus ainda não foi implementado.
            ValueError: O valor de alfa deve ser 30, 60 ou 90.

        Returns:
            StrBooleanDict: O sinal de trip.
        """
        match self._alpha:
            case 30:
                raise NotImplementedError("Relé 32 de fase com alfa igual a 30 graus ainda não foi implementado.")
            case 60:
                raise NotImplementedError("Relé 32 de fase com alfa igual a 60 graus ainda não foi implementado.")
            case 90:
                v_pol_phase = {
                    'a': np.angle(self._phasors['vb'] - self._phasors['vc'], deg=True),  # type: ignore
                    'b': np.angle(self._phasors['vc'] - self._phasors['va'], deg=True),  # type: ignore
                    'c': np.angle(self._phasors['va'] - self._phasors['vb'], deg=True)  # type: ignore
                }
                i_op_phase = {
                    phase_name[1:]: np.angle(current, deg=True)  # type: ignore
                    for phase_name, current in self._phasors.get_currents()
                }
                op_region = {
                    name: (
                        v_pol_phase[name] - self._alpha + self._beta,
                        v_pol_phase[name] + self._alpha + self._beta
                    )
                    for name in ['a', 'b', 'c']
                }

                trip_permission: dict[str, npt.NDArray[np.bool_]] = {}
                for phase in ['a', 'b', 'c']:
                    in_region = (i_op_phase[phase] > op_region[phase][0]) & (i_op_phase[phase] < op_region[phase][1])
                    avg_result = np.round(np.mean(in_region), 0).astype(bool)
                    trip_permission[phase] = np.full_like(i_op_phase[phase], avg_result, dtype=bool)
                return trip_permission

            case _:
                raise ValueError("O valor de alfa deve ser 30, 60 ou 90.")


class NeutralRelay32(_Relay32Base):
    def __init__(
        self,
        alpha: float,
        beta: float,
        zero_seq_voltage: numeric_array,
        zero_seq_current: numeric_array,
    ) -> None:
        """Implementa a lógica de atuação do relé 32 de neutro.

        Args:
            alpha (float): O ângulo alfa do relé.
            beta (float): O ângulo beta do relé.
            zero_seq_voltage (numeric_array): As tensões de sequência zero.
            zero_seq_current (numeric_array): As correntes de sequência zero.
        """
        super().__init__(alpha, beta)
        self._zero_seq_voltage = zero_seq_voltage
        self._zero_seq_current = zero_seq_current

    def analyze_trip(self) -> StrBooleanDict:
        """Analisa o sinal de trip do relé 32 de neutro.

        Raises:
            NotImplementedError: Ângulo alfa igual a 30 ou 60 graus ainda não foi implementado.
            ValueError: O valor de alfa deve ser 30, 60 ou 90.

        Returns:
            StrBooleanDict: O sinal de trip.
        """
        match self._alpha:
            case 30:
                raise NotImplementedError("Relé 32 de neutro com alfa igual a 30 graus ainda não foi implementado.")
            case 60:
                raise NotImplementedError("Relé 32 de neutro com alfa igual a 60 graus ainda não foi implementado.")
            case 90:
                v_pol_phase = np.angle(self._zero_seq_voltage, deg=True)  # type: ignore
                i_op_phase = np.angle(self._zero_seq_current, deg=True)  # type: ignore
                op_region = (v_pol_phase - self._alpha + self._beta, v_pol_phase + self._alpha + self._beta)
                in_region = (i_op_phase > op_region[0]) & (i_op_phase < op_region[1])

                avg_result = np.round(np.mean(in_region), 0).astype(bool)
                trip_permission = np.full_like(i_op_phase, avg_result, dtype=bool)
                return {'neutral': trip_permission}

            case _:
                raise ValueError("O valor de alfa deve ser 30, 60 ou 90.")


class _OvercurrentRelayBase(ABC):
    def __init__(self, adjust_current: float, time_vector: numeric_array) -> None:
        """Classe abstrata para relés de sobrecorrente.

        Args:
            adjust_current (float): A corrente de ajuste do relé.
            time_vector (numeric_array): O vetor de tempo.
        """
        self._adjust_current = adjust_current
        self._time_vector = time_vector

    @abstractmethod
    def _calculate_trip_time(self, secondary_current: numeric_array) -> numeric_array:
        """Método abstrato para cálculo do tempo de atuação do relé.

        Args:
            secondary_current (numeric_array): Os valores de corrente já ajustados para o secundário do TC.

        Raises:
            NotImplementedError: Deve ser implementado nas subclasses.

        Returns:
            numeric_array: O tempo de atuação do relé.
        """
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")

    @abstractmethod
    def analyze_trip(self) -> dict[str, number]:
        """Método abstrato para análise de trip do relé.

        Raises:
            NotImplementedError: Deve ser implementado nas subclasses.

        Returns:
            dict[str, number]: O menor tempo de atuação do relé para cada fase.
        """
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")


class _Relay50Base(_OvercurrentRelayBase):
    def __init__(self, adjust_current: float, time_vector: numeric_array) -> None:
        """Classe abstrata para relés de sobrecorrente temporizados.

        Args:
            adjust_current (float): A corrente de ajuste do relé 50 (referida ao secundário do TC).
            time_vector (numeric_array): O vetor de tempo.
        """
        super().__init__(adjust_current, time_vector)

    def _calculate_trip_time(self, secondary_current: numeric_array) -> numeric_array:
        """Calcula o tempo de atuação do relé 50 para cada par corrente-tempo.

        Args:
            secondary_current (numeric_array): Os valores de corrente já ajustados para o secundário do TC.

        Returns:
            numeric_array: O tempo de atuação do relé 50.
        """
        trip_times = np.where(
            secondary_current <= self._adjust_current,
            np.inf,
            self._time_vector,
        )
        return trip_times


class PhaseRelay50(_Relay50Base):
    def __init__(self, adjust_current: float, time_vector: numeric_array, current_phasors: Signals) -> None:
        """Implementa a lógica de atuação do relé 50 para cada fase.

        Args:
            adjust_current (float): A corrente de ajuste do relé 50 (referida ao secundário do TC).
            time_vector (numeric_array): O vetor de tempo.
            current_phasors (Signals): O conjunto trifásico de sinais de corrente.
        """
        super().__init__(adjust_current, time_vector)
        self._phasors = current_phasors

    def analyze_trip(self) -> dict[str, number]:
        """Calcula o tempo de atuação do relé 50 para cada fase."""
        trip_times = {
            current_name: self._calculate_trip_time(secondary_current=np.abs(current_value))
            for current_name, current_value in self._phasors.get_currents()
        }

        min_trip_times = {current: np.min(times) for current, times in trip_times.items()}
        return min_trip_times


class NeutralRelay50(_Relay50Base):
    def __init__(self, adjust_current: float, time_vector: numeric_array, neutral_current: numeric_array) -> None:
        """Implementa a lógica de atuação do relé 50 para a corrente de neutro.

        Args:
            adjust_current (float): A corrente de ajuste do relé 50 (referida ao secundário do TC).
            time_vector (numeric_array): O vetor de tempo.
            neutral_current (numeric_array): A corrente de neutro.
        """
        super().__init__(adjust_current, time_vector)
        self._neutral_current = neutral_current

    def analyze_trip(self) -> dict[str, number]:
        """Calcula o tempo de atuação do relé 50 para a corrente de neutro."""
        trip_time = self._calculate_trip_time(secondary_current=np.abs(self._neutral_current))
        return {"neutral": np.min(trip_time)}


class _Relay51Base(_OvercurrentRelayBase):
    ADJUST_OFFSET = 0.01

    def __init__(self, gamma: float, adjust_current: float, curve: str, time_vector: numeric_array) -> None:
        """Classe abstrata para relés de sobrecorrente temporizados.

        Args:
            gamma (float): O multiplicador do tempo de ajuste do relé 51.
            adjust_current (float): A corrente de ajuste do relé 51 (referida ao secundário do TC).
            curve (str): A curva do relé 51.
            time_vector (numeric_array): O vetor de tempo.
        """
        super().__init__(adjust_current, time_vector)
        self._gamma = gamma
        self._k, self._a, self._c = Curve[curve.upper()].value

    def _calculate_trip_time(self, secondary_current: numeric_array) -> numeric_array:
        """Calcula o tempo de atuação do relé 51 para cada par corrente-tempo.

        Args:
            secondary_current (numeric_array): Os valores de corrente já ajustados para o secundário do TC.

        Returns:
            numeric_array: O tempo de atuação do relé 51.
        """
        normalized_current = np.where(
            secondary_current == self._adjust_current,
            self._adjust_current + _Relay51Base.ADJUST_OFFSET,
            secondary_current,
        ) / self._adjust_current

        curve_common_term = self._gamma * (self._k / ((normalized_current) ** self._a - 1) + self._c)

        trip_times = np.where(
            secondary_current <= self._adjust_current,
            np.inf,
            self._time_vector + curve_common_term,
        )
        return trip_times

    @abstractmethod
    def analyze_trip(self) -> dict[str, np.number]:
        """Método abstrato para análise de trip do relé 51.

        Raises:
            NotImplementedError: Deve ser implementado nas subclasses.

        Returns:
            dict[str, np.number]: O menor tempo de atuação do relé para cada fase.
        """
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")


class PhaseRelay51(_Relay51Base):
    def __init__(
        self,
        gamma: float,
        adjust_current: float,
        curve: str,
        time_vector: numeric_array,
        current_phasors: Signals,
    ) -> None:
        """Implementa a lógica de atuação do relé 51 para cada fase.

        Args:
            gamma (float): O multiplicador do tempo de ajuste do relé 51.
            adjust_current (float): A corrente de ajuste do relé 51 (referida ao secundário do TC).
            curve (str): A curva do relé 51.
            time_vector (numeric_array): O vetor de tempo.
            current_phasors (Signals): O conjunto trifásico de sinais de corrente.
        """
        super().__init__(gamma, adjust_current, curve, time_vector)
        self._phasors = current_phasors

    def analyze_trip(self) -> dict[str, number]:
        """Calcula o tempo de atuação do relé 51 para cada fase."""
        trip_times = {
            current_name: self._calculate_trip_time(secondary_current=np.abs(current_value))
            for current_name, current_value in self._phasors.get_currents()
        }
        min_trip_times = {name: np.min(times) for name, times in trip_times.items()}
        return min_trip_times


class NeutralRelay51(_Relay51Base):
    def __init__(
        self,
        gamma: float,
        adjust_current: float,
        curve: str,
        time_vector: numeric_array,
        neutral_current: numeric_array,
    ) -> None:
        """Implementa a lógica de atuação do relé 51 para a corrente de neutro.

        Args:
            gamma (float): O multiplicador do tempo de ajuste do relé 51.
            adjust_current (float): A corrente de ajuste do relé 51 (referida ao secundário do TC).
            curve (str): A curva do relé 51.
            time_vector (numeric_array): O vetor de tempo.
            neutral_current (numeric_array): A corrente de neutro.
        """
        super().__init__(gamma, adjust_current, curve, time_vector)
        self._neutral_current = neutral_current

    def analyze_trip(self) -> dict[str, number]:
        """Calcula o tempo de atuação do relé 51 para a corrente de neutro."""
        trip_time = self._calculate_trip_time(secondary_current=np.abs(self._neutral_current))
        return {"neutral": np.min(trip_time)}


class _Relay67Base(ABC):
    def __init__(
        self,
        trips: dict[str, dict[str, np.float64 | np.bool_]],
        time_vector: numeric_array
    ) -> None:
        """Classe abstrata para relés 67.

        Args:
            trips (dict[str, dict[str, np.float64  |  np.bool_]]): Sinais de trip dos relés 50, 51 e 32.
            time_vector (numeric_array): O vetor de tempo.
        """
        self._trips = trips
        self._time_vector = time_vector

    def _validate_trips(self, relay_types: list[str]) -> None:
        """Valida se os relés 50, 51 e 32 estão presentes no dicionário de sinais de trip.

        Args:
            relay_types (list[str]): Os tipos de relés a serem validados.

        Raises:
            ValueError: Se algum relé não estiver presente no dicionário de sinais de trip.
        """
        missing_relays = [relay for relay in relay_types if relay not in self._trips.keys()]
        if missing_relays:
            raise ValueError(f"Os relés {missing_relays} não foram encontrados. Por favor, adiciones-os ao IED.")

    @abstractmethod
    def analyze_trip(self) -> StrBooleanDict:
        """Método abstrato para análise de trip do relé 67.

        Raises:
            NotImplementedError: Deve ser implementado nas subclasses.

        Returns:
            StrBooleanDict: O sinal de trip.
        """
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")


class PhaseRelay67(_Relay67Base):
    def __init__(
        self,
        trips: dict[str, dict[str, np.float64 | np.bool_]],
        time_vector: numeric_array
    ) -> None:
        """Implementa a lógica de atuação do relé 67 de fase.

        Args:
            trips (dict[str, dict[str, np.float64  |  np.bool_]]): Sinais de trip dos relés 50, 51 e 32.
            time_vector (numeric_array): O vetor de tempo.
        """
        super().__init__(trips, time_vector)
        self.__validate_trips()

    def __validate_trips(self) -> None:
        """Valida se os relés 50, 51 e 32 de fase estão presentes no dicionário de sinais de trip."""
        required_relays = ["50F", "51F", "32F"]
        super()._validate_trips(required_relays)

    def analyze_trip(self) -> StrBooleanDict:
        """Analisa o sinal de trip do relé 67 de fase.

        Returns:
            StrBooleanDict: O sinal de trip.
        """
        phase_map = {"ia": "a", "ib": "b", "ic": "c"}
        state_50F = {phase: self._time_vector >= self._trips["50F"][phase] for phase in ["ia", "ib", "ic"]}
        state_51F = {phase: self._time_vector >= self._trips["51F"][phase] for phase in ["ia", "ib", "ic"]}

        trip_signal = {
            phase: (
                (state_50F[phase] & self._trips["32F"][phase_map[phase]].astype(bool))
                | (state_51F[phase] & self._trips["32F"][phase_map[phase]].astype(bool))
            )
            for phase in ["ia", "ib", "ic"]
        }
        return trip_signal


class NeutralRelay67(_Relay67Base):
    def __init__(
        self,
        trips: dict[str, dict[str, np.float64 | np.bool_]],
        time_vector: numeric_array
    ) -> None:
        """Implementa a lógica de atuação do relé 67 de neutro.

        Args:
            trips (dict[str, dict[str, np.float64  |  np.bool_]]): Sinais de trip dos relés 50, 51 e 32.
            time_vector (numeric_array): O vetor de tempo.
        """
        super().__init__(trips, time_vector)
        self.__validate_trips()

    def __validate_trips(self) -> None:
        """Valida se os relés 50, 51 e 32 de neutro estão presentes no dicionário de sinais de trip."""
        required_relays = ["50N", "51N", "32N"]
        super()._validate_trips(required_relays)

    def analyze_trip(self) -> StrBooleanDict:
        """Analisa o sinal de trip do relé 67 de neutro.

        Returns:
            StrBooleanDict: O sinal de trip.
        """

        state_50N = self._time_vector >= self._trips["50N"]["neutral"]
        state_51N = self._time_vector >= self._trips["51N"]["neutral"]
        trip_signal = (
            (state_50N & self._trips["32N"]["neutral"].astype(np.bool_)) |  # casting to avoid mypy warning
            (state_51N & self._trips["32N"]["neutral"].astype(np.bool_))
        )
        return {"neutral": trip_signal}
