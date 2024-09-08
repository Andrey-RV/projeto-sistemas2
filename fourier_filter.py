import numpy as np


class FourierFilter:
    def __init__(self, samples_per_cycle):
        '''
        Instancia um filtro de fourier contendo os filtros cosseno e seno.
        Attributes:
            samples_per_cycle: a quantidade de amostras de sinal por ciclo do sinal.
        '''
        self.samples_per_cycle = samples_per_cycle
        self.create_filter()

    def create_filter(self):
        '''
        Cria os filtros cosseno e seno de Fourier de período completo
        '''
        # Gera N argumentos espaçados entre (2.k.pi/N) e 2.pi
        filter_args = np.linspace((2*np.pi / self.samples_per_cycle), 2*np.pi, self.samples_per_cycle)
        self.cosine_filter = ((2 / self.samples_per_cycle) * np.cos(filter_args))
        self.sine_filter = ((2 / self.samples_per_cycle) * np.sin(filter_args))


class PhasorEstimator:
    def __init__(self, signal, samples_per_cycle):
        '''
        Instancia um estimador de fasor.
        Attributes:
            signal: Uma sequência de amostras que compõe o sinal.
            samples_per_cycle: a quantidade de amostras de sinal por ciclo do sinal.
        '''
        self.signal = signal.reshape(-1,)
        self.fourier_filter = FourierFilter(samples_per_cycle)

    def estimate(self):
        '''
        Estima um fasor utilizando a convolução do sinal com os filtros de Fourier cosseno e seno.
        '''
        self.real_part = np.convolve(self.signal, self.fourier_filter.cosine_filter)
        self.imaginary_part = np.convolve(self.signal, self.fourier_filter.sine_filter)
        self.amplitude = np.sqrt(self.real_part**2 + self.imaginary_part**2)
        phase_rad = np.arctan2(self.imaginary_part, self.real_part)
        self.phase = np.degrees(phase_rad)
