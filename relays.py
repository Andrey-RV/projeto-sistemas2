import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from signals import Signals
from curves import Curve


class Relay21:
    ...


class _Relay32Base(ABC):
    def __init__(self, alpha: float, beta: float) -> None:
        self._alpha = alpha
        self._beta = beta

    @abstractmethod
    def analyze_trip(self) -> dict[str, npt.NDArray[np.bool_]]:
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")

    def _trim_data(self, array, num_points_to_be_chopped):
        '''Substitui n amostras antes e n/2 amostras após uma transição + -> - por NaN.'''
        positive_values = (array[:-1] >= 0)
        negative_values = (array[1:] < 0)
        transitions = np.where(positive_values & negative_values)[0] + 1

        for transition_index in transitions:
            start_index = max(transition_index - num_points_to_be_chopped, 0)  # Include points before
            end_index = min(transition_index + num_points_to_be_chopped // 2 + 1, len(array))  # Include points after
            array[start_index:end_index] = np.nan

        return array


class PhaseRelay32(_Relay32Base):
    def __init__(
        self,
        alpha: float,
        beta: float,
        phasors: Signals,
    ) -> None:

        super().__init__(alpha, beta)
        self._phasors = phasors

    def analyze_trip(self) -> dict[str, npt.NDArray[np.bool_]]:
        match self._alpha:
            case 30:
                raise NotImplementedError("Relé 32 de fase com alfa igual a 30 graus ainda não foi implementado.")
            case 60:
                raise NotImplementedError("Relé 32 de fase com alfa igual a 60 graus ainda não foi implementado.")
            case 90:
                v_pol_phase = {
                    'a': np.angle(self._phasors['vb'] - self._phasors['vc'], deg=True),
                    'b': np.angle(self._phasors['vc'] - self._phasors['va'], deg=True),
                    'c': np.angle(self._phasors['va'] - self._phasors['vb'], deg=True)
                }
                i_op_phase = {
                    phase_name[1:]: np.angle(current, deg=True)  # ix -> x
                    for phase_name, current in self._phasors.get_currents()
                }
                op_region = {
                    name: (
                        v_pol_phase[name] - self._alpha + self._beta,
                        v_pol_phase[name] + self._alpha + self._beta
                    )
                    for name in ['a', 'b', 'c']
                }
                trip_permission = {
                    name: np.round(np.mean(
                        ((i_op_phase[name] > op_region[name][0]) | np.isnan(i_op_phase[name])) &
                        ((i_op_phase[name] < op_region[name][1]) | np.isnan(i_op_phase[name]))
                    ), 0).astype(np.bool_)
                    for name in ['a', 'b', 'c']
                }

                trip_permission = {
                    name: np.full_like(i_op_phase[name], trip_permission[name], dtype=np.bool_)
                    for name in ['a', 'b', 'c']
                }

                return trip_permission
            case _:
                raise ValueError("O valor de alfa deve ser 30, 60 ou 90.")


class NeutralRelay32(_Relay32Base):
    def __init__(
        self,
        alpha: float,
        beta: float,
        zero_seq_voltage: npt.NDArray[np.complex128],
        zero_seq_current: npt.NDArray[np.complex128],
    ) -> None:

        super().__init__(alpha, beta)
        self._zero_seq_voltage = zero_seq_voltage
        self._zero_seq_current = zero_seq_current

    def analyze_trip(self) -> dict[str, npt.NDArray[np.bool_]]:
        match self._alpha:
            case 30:
                raise NotImplementedError("Relé 32 de neutro com alfa igual a 30 graus ainda não foi implementado.")
            case 60:
                raise NotImplementedError("Relé 32 de neutro com alfa igual a 60 graus ainda não foi implementado.")
            case 90:
                v_pol_phase = np.angle(self._zero_seq_voltage, deg=True)
                i_op_phase = np.angle(self._zero_seq_current, deg=True)
                op_region = (-v_pol_phase - self._alpha + self._beta, -v_pol_phase + self._alpha + self._beta)

                trip_permission = np.round(np.mean(
                    ((i_op_phase > op_region[0]) | np.isnan(i_op_phase)) &  # Está acima do limite inferior ou é NaN
                    ((i_op_phase < op_region[1]) | np.isnan(i_op_phase))    # Está abaixo do limite superior ou é NaN
                ), 0).astype(np.bool_)

                trip_permission = np.full_like(i_op_phase, trip_permission, dtype=np.bool_)

                return {'neutral': trip_permission}

            case _:
                raise ValueError("O valor de alfa deve ser 30, 60 ou 90.")


class _OvercurrentRelayBase(ABC):
    def __init__(self, adjust_current: float, time_vector: npt.NDArray[np.float64]) -> None:
        """Classe abstrata para relés de sobrecorrente.

        Args:
            adjust_current (float): A corrente de ajuste do relé.
            time_vector (npt.NDArray[np.float64]): O vetor de tempo.
        """
        self._adjust_current = adjust_current
        self._time_vector = time_vector

    @abstractmethod
    def _calculate_trip_time(self, secondary_current: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")

    @abstractmethod
    def analyze_trip(self) -> dict[str, np.float64]:
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")


class _Relay50Base(_OvercurrentRelayBase):
    def __init__(self, adjust_current: float, time_vector: npt.NDArray[np.float64]) -> None:
        """Classe abstrata para relés de sobrecorrente temporizados.

        Args:
            adjust_current (float): A corrente de ajuste do relé 50 (referida ao secundário do TC).
            time_vector (npt.NDArray[np.float64]): O vetor de tempo.
        """
        super().__init__(adjust_current, time_vector)

    def _calculate_trip_time(self, secondary_current: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
        """Calcula o tempo de atuação do relé 50 para cada par corrente-tempo.

        Args:
            secondary_current (npt.NDArray[np.complex128]): Os valores de corrente já ajustados para o secundário do TC.

        Returns:
            npt.NDArray[np.float64]: O tempo de atuação do relé 50.
        """
        trip_times = np.where(
            secondary_current <= self._adjust_current,
            np.inf,
            self._time_vector,
        )
        return trip_times


class PhaseRelay50(_Relay50Base):
    def __init__(self, adjust_current: float, time_vector: npt.NDArray[np.float64], current_phasors: Signals) -> None:
        super().__init__(adjust_current, time_vector)
        self._phasors = current_phasors

    def analyze_trip(self) -> dict[str, np.float64]:
        """Calcula o tempo de atuação do relé 50 para cada fase."""
        trip_times = {
            current_name: self._calculate_trip_time(secondary_current=np.abs(current_value))
            for current_name, current_value in self._phasors.get_currents()
        }

        min_trip_times = {current: np.min(times) for current, times in trip_times.items()}
        return min_trip_times


class NeutralRelay50(_Relay50Base):
    def __init__(
        self,
        adjust_current: float,
        time_vector: npt.NDArray[np.float64],
        neutral_current: npt.NDArray[np.complex128]
    ) -> None:
        super().__init__(adjust_current, time_vector)
        self._neutral_current = neutral_current

    def analyze_trip(self) -> dict[str, np.float64]:
        """Calcula o tempo de atuação do relé 50 para a corrente de neutro."""
        trip_times = {
            "neutral": self._calculate_trip_time(secondary_current=np.abs(self._neutral_current))
        }
        min_trip_time = {'neutral': np.min(trip_times["neutral"])}
        return min_trip_time


class _Relay51Base(_OvercurrentRelayBase):
    ADJUST_OFFSET = 0.01

    def __init__(self, gamma: float, adjust_current: float, curve: str, time_vector: npt.NDArray[np.float64]) -> None:
        """Classe abstrata para relés de sobrecorrente temporizados.

        Args:
            gamma (float): O multiplicador do tempo de ajuste do relé 51.
            adjust_current (float): A corrente de ajuste do relé 51 (referida ao secundário do TC).
            curve (str): A curva do relé 51.
            time_vector (npt.NDArray[np.float64]): O vetor de tempo.
        """
        super().__init__(adjust_current, time_vector)
        self._gamma = gamma
        self._k, self._a, self._c = Curve[curve.upper()].value

    def _calculate_trip_time(self, secondary_current: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
        """Calcula o tempo de atuação do relé 51 para cada par corrente-tempo.

        Args:
            secondary_current (npt.NDArray[np.complex128]): Os valores de corrente já ajustados para o secundário do TC.

        Returns:
            npt.NDArray[np.float64]: O tempo de atuação do relé 51.
        """
        normalized_current = (np.where(  # Evita divisão por zero + normalização
            secondary_current == self._adjust_current,
            self._adjust_current + _Relay51Base.ADJUST_OFFSET,
            secondary_current)
        ) / self._adjust_current

        curve_common_term = self._gamma * (self._k / ((normalized_current) ** self._a - 1) + self._c)

        trip_times = np.where(
            secondary_current <= self._adjust_current,
            np.inf,
            self._time_vector + curve_common_term,
        )
        return trip_times

    @abstractmethod
    def analyze_trip(self) -> dict[str, np.float64]:
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")


class PhaseRelay51(_Relay51Base):
    def __init__(
        self,
        gamma: float,
        adjust_current: float,
        curve: str,
        time_vector: npt.NDArray[np.float64],
        current_phasors: Signals,
    ) -> None:

        super().__init__(gamma, adjust_current, curve, time_vector)
        self._phasors = current_phasors

    def analyze_trip(self) -> dict[str, np.float64]:
        """Calcula o tempo de atuação do relé 51 para cada fase."""
        trip_times = {
            current_name: self._calculate_trip_time(secondary_current=np.abs(current_value))
            for current_name, current_value in self._phasors.get_currents()
        }
        min_trip_times = {current: np.min(times) for current, times in trip_times.items()}
        return min_trip_times


class NeutralRelay51(_Relay51Base):
    def __init__(
        self,
        gamma: float,
        adjust_current: float,
        curve: str,
        time_vector: npt.NDArray[np.float64],
        neutral_current: npt.NDArray[np.complex128],
    ) -> None:

        super().__init__(gamma, adjust_current, curve, time_vector)
        self._neutral_current = neutral_current

    def analyze_trip(self) -> dict[str, np.float64]:
        """Calcula o tempo de atuação do relé 51 para a corrente de neutro."""
        trip_times = {
            "neutral": self._calculate_trip_time(secondary_current=np.abs(self._neutral_current))
        }
        min_trip_time = {'neutral': np.min(trip_times["neutral"])}
        return min_trip_time


class _Relay67Base(ABC):
    def __init__(
        self,
        trips: dict[str, dict[str, np.float64 | np.bool_]],
        time_vector: npt.NDArray[np.float64]
    ) -> None:
        self._trips = trips
        self._time_vector = time_vector

    def _validate_trips(self, relay_types: list[str]) -> None:
        missing_relays = [relay_type for relay_type in relay_types if relay_type not in self._trips.keys()]
        if missing_relays:
            raise ValueError(f"Os relés {missing_relays} não foram encontrados. Por favor, adiciones-os ao IED.")

    @abstractmethod
    def analyze_trip(self) -> dict[str, npt.NDArray[np.bool_]]:
        raise NotImplementedError("Esse método deve ser implementado nas subclasses.")


class PhaseRelay67(_Relay67Base):
    def __init__(
        self,
        trips: dict[str, dict[str, np.float64 | np.bool_]],
        time_vector: npt.NDArray[np.float64]
    ) -> None:
        super().__init__(trips, time_vector)
        self.__validate_trips()

    def __validate_trips(self) -> None:
        relay_types = ["50F", "51F", "32F"]
        super()._validate_trips(relay_types)

    def analyze_trip(self) -> dict[str, npt.NDArray[np.bool_]]:
        state_50F = {
            phase: (self._time_vector >= self._trips["50F"][phase])
            for phase in ["ia", "ib", "ic"]
        }
        state_51F = {
            phase: (self._time_vector >= self._trips["51F"][phase])
            for phase in ["ia", "ib", "ic"]
        }
        trip_signal = {
            phase: (
                (state_50F[phase] & self._trips["32F"][phase[1:]].astype(np.bool_)) |  # casting to avoid mypy warning
                (state_51F[phase] & self._trips["32F"][phase[1:]].astype(np.bool_))
            )
            for phase in ["ia", "ib", "ic"]
        }
        return trip_signal


class NeutralRelay67(_Relay67Base):
    def __init__(
        self,
        trips: dict[str, dict[str, np.float64 | np.bool_]],
        time_vector: npt.NDArray[np.float64]
    ) -> None:
        super().__init__(trips, time_vector)
        self.__validate_trips()

    def __validate_trips(self) -> None:
        relay_types = ["50N", "51N", "32N"]
        super()._validate_trips(relay_types)

    def analyze_trip(self) -> dict[str, npt.NDArray[np.bool_]]:
        state_50N = self._time_vector >= self._trips["50N"]["neutral"]
        state_51N = self._time_vector >= self._trips["51N"]["neutral"]
        trip_signal = (
            (state_50N & self._trips["32N"]["neutral"].astype(np.bool_)) |  # casting to avoid mypy warning
            (state_51N & self._trips["32N"]["neutral"].astype(np.bool_))
        )
        return {"neutral": trip_signal}
