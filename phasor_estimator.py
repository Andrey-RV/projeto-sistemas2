from scipy.signal import fftconvolve
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

        estimated_phasor = real + 1j * imaginary
        return estimated_phasor
