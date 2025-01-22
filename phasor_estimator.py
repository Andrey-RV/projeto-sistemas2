from scipy.signal import fftconvolve  # type: ignore
from filters import FourierFilter
import numpy as np
import numpy.typing as npt


class PhasorEstimator:
    def __init__(self, signal: npt.NDArray[np.float64], samples_per_cycle: int) -> None:
        '''
        Instancia um estimador de fasor.
        Args:
            signal (npt.NDArray[np.float64]): o sinal a ser estimado.
            samples_per_cycle (int): A quantidade de amostras capturadas pelo IED em um período da onda fundamental.

        Returns:
            None
        '''
        self.__signal = np.array(signal)
        self.__samples_per_cycle = samples_per_cycle
        self.__fourier_filters = FourierFilter(self.__samples_per_cycle)

    def __repr__(self, verbose: bool = False) -> str:
        """Retorna uma representação do objeto em forma de string.

        Args:
            verbose (bool, optional): Seleciona se a representação conterá somente o shape e dtype do sinal ou parte do sinal em si. Defaults to False.

        Returns:
            str: Representação do objeto em forma de string.
        """
        if verbose:
            signal_repr = f"signal={np.array2string(self.__signal, precision=3, threshold=5)}"
        else:
            signal_repr = f"signal=ndarray(shape={self.__signal.shape}, dtype={self.__signal.dtype})"
        return (
            f"PhasorEstimator("
            f"{signal_repr}, "
            f"samples_per_cycle={self.__samples_per_cycle}, "
            f"amplitude={'set' if hasattr(self, 'amplitude') else 'unset'}, "
            f"phase={'set' if hasattr(self, 'phase') else 'unset'})"
        )

    def estimate(self) -> None:
        '''
        Estima um fasor utilizando a convolução do sinal com os filtros de Fourier cosseno e seno.

        Returns:
            None
        '''
        self.__fourier_filters.create_filter()
        real = fftconvolve(self.__signal, self.__fourier_filters.cosine_filter, mode='same')
        imaginary = fftconvolve(self.__signal, self.__fourier_filters.sine_filter, mode='same')
        rect_form = real + 1j * imaginary
        self.amplitude = np.abs(rect_form)
        self.phase = np.degrees(np.angle(rect_form))
        self.exp_form = self.amplitude * np.exp(1j * np.radians(self.phase))
