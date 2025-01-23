from scipy.signal import fftconvolve  # type: ignore
from filters import FourierFilter
import numpy as np
import numpy.typing as npt


class PhasorEstimator:
    def __init__(self, samples_per_cycle: int) -> None:
        '''
        Instancia um estimador de fasor.
        Args:
            signal (npt.NDArray[np.float64]): o sinal a ser estimado.
            samples_per_cycle (int): A quantidade de amostras capturadas pelo IED em um período da onda fundamental.

        Returns:
            None
        '''
        self.__samples_per_cycle = samples_per_cycle
        self.__fourier_filters = FourierFilter(self.__samples_per_cycle)

    def __repr__(self) -> str:
        return (
            f"PhasorEstimator("
            f"samples_per_cycle={self.__samples_per_cycle})"
        )

    def estimate(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        '''
        Retorna a representação complexa do fasor estimado.

        Returns:
            complex: O fasor estimado.
        '''
        self.__fourier_filters.create_filter()
        real = fftconvolve(signal, self.__fourier_filters.cosine_filter, mode='same')
        imaginary = fftconvolve(signal, self.__fourier_filters.sine_filter, mode='same')

        phasor = real + 1j * imaginary
        return phasor
