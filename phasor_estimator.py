from typing import Sequence
from scipy.signal import fftconvolve  # type: ignore
from filters import FourierFilter
import numpy as np


class PhasorEstimator:
    def __init__(self, signal: Sequence[float], sample_rate: float, num_points: int) -> None:
        '''
        Instancia um estimador de fasor.
        Args:
            signal (Sequence[float]): o sinal a ser estimado.
            sample_rate (float): a taxa de amostragem do sinal.
        '''
        self.signal = np.array(signal).reshape(-1,)
        self.fourier_filters = FourierFilter(sample_rate)
        self.num_points = num_points

    def estimate(self) -> None:
        '''
        Estima um fasor utilizando a convolução do sinal com os filtros de Fourier cosseno e seno.
        '''
        real = fftconvolve(self.signal, self.fourier_filters.cosine_filter, mode='same')
        imaginary = fftconvolve(self.signal, self.fourier_filters.sine_filter, mode='same')
        rect_form = real + 1j * imaginary
        self.amplitude = np.abs(rect_form)
        self.phase = np.degrees(np.angle(rect_form))
        self.exp_form = self.amplitude * np.exp(1j * np.radians(self.phase))
