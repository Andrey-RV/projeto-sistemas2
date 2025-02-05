from scipy.signal import fftconvolve  # type: ignore
from filters import FourierFilter
import numpy as np
import numpy.typing as npt


class PhasorEstimator:
    def __init__(self, samples_per_cycle: int) -> None:
        '''
        Instancia um estimador de fasor.

        Args:
            samples_per_cycle (int): A quantidade de amostras capturadas pelo IED em um período da onda fundamental.
        '''
        self.__samples_per_cycle = samples_per_cycle
        self.__fourier_filters = FourierFilter(self.__samples_per_cycle)
        self.__fourier_filters.create_filter()

    def __repr__(self) -> str:
        return (f"PhasorEstimator(samples_per_cycle={self.__samples_per_cycle})")

    def estimate(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        '''
        Retorna a representação complexa do fasor estimado.

        Args:
            signal (npt.NDArray[np.float64]): O sinal base para a estimativa do fasor.

        Returns:
            npt.NDArray[np.complex128]: O fasor estimado.
        '''

        real = fftconvolve(signal, self.__fourier_filters.cosine_filter)[:len(signal)]
        imaginary = fftconvolve(signal, self.__fourier_filters.sine_filter)[:len(signal)]

        phasor = real + 1j * imaginary
        return phasor
