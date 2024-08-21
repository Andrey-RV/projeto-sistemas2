import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FourierFilter:
    def __init__(self, sample_rate, is_full_period=True):
        '''
        Instancializa um filtro de Fourier
        Args:
            sample_rate: taxa de amostragem do sinal de entrada
            is_full_period: se é um filtro de período completo ou de meio período
        '''
        self.sample_rate = sample_rate
        self.is_full_period = is_full_period
        self.create_filter()

    def create_filter(self):
        '''
            Cria os filtros cosseno e seno
        '''
        if self.is_full_period:
            filter_args = np.linspace((2*np.pi / self.sample_rate), 2*np.pi, self.sample_rate)
            self.cosine_filter = ((2 / self.sample_rate) * np.cos(filter_args)).round(4)
            self.sine_filter = ((2 / self.sample_rate) * np.sin(filter_args)).round(4)
        else:
            filter_args = np.linspace((2*np.pi / self.sample_rate), np.pi, (self.sample_rate / 2))
            self.cosine_filter = ((4 / self.sample_rate) * np.cos(filter_args)).round(4)
            self.sine_filter = ((4 / self.sample_rate) * np.sin(filter_args)).round(4)


class PhasorEstimator:
    def __init__(self, signal, sample_rate, is_full_period=True):
        '''
        Instancializa um estimador de fasor utilizando filtros de Fourier
        Args:
            signal: sinal de entrada
            sample_rate: taxa de amostragem do sinal de entrada
            is_full_period: se a estimação será feito com um filtro de período completo ou de meio período
        '''
        self.signal = signal
        self.sample_rate = sample_rate
        self.is_full_period = is_full_period
        self.fourier_filter = FourierFilter(sample_rate, is_full_period)

    def estimate(self):
        '''
        Estima o fasor do sinal de entrada através da convolução do sinal com os filtros de Fourier
        '''
        self.real_part = np.convolve(self.signal, self.fourier_filter.cosine_filter).round(4)
        self.imaginary_part = np.convolve(self.signal, self.fourier_filter.sine_filter).round(4)
        self.amplitude = np.sqrt(self.real_part**2 + self.imaginary_part**2).round(4)
        phase_rad = np.arctan2(self.imaginary_part, self.real_part)
        self.phase = np.degrees(phase_rad).round(4)


signal = pd.read_csv("./sinal.dat", delimiter='\t', names=['t', 'x'])
phasor = PhasorEstimator(signal=signal['x'], sample_rate=12, is_full_period=True)
phasor.estimate()

plt.plot(signal['t'], signal['x'], '-o', label='Sinal')
plt.plot(signal['t'], phasor.amplitude[:61], '-o', label="Amplitude do fasor estimado")
plt.legend(prop={'size': 9})
plt.show()
plt.plot(signal['t'], phasor.amplitude[:61])
plt.show()
plt.plot(signal['t'], phasor.phase[0:61])
plt.show()