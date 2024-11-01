import numpy as np


class AntiAliasingFilter():
    def __init__(self, period, signal, b, c):
        '''
        Instancia um filtro de anti-aliasing.
        Args:
            signal: Uma sequência de amostras que compõe o sinal.
            period: A diferença de tempo entre as amostras.
            b: Parâmetro b
            c: Parâmetro c
        '''
        self.signal = signal
        self.period = period
        self.a = 2 / self.period
        self.b = b
        self.c = c
        self.apply_filter()

    def apply_filter(self):
        '''
            Aplica o filtro anti-aliasing ao sinal de entrada via
            xout(n) = [c0.xin(n) + c1.xin(n-1) + c2.xin(n-2)] - [d1.xout(n-1) + d2.xout(n-2)]
        '''
        c0 = self.c / (self.a**2 + self.a*self.b + self.c)
        c1 = 2 * c0
        c2 = c0
        d1 = 2 * (self.c - self.a**2) / (self.a**2 + self.a*self.b + self.c)
        d2 = (self.a**2 + self.c - self.a*self.b) / (self.a**2 + self.a*self.b + self.c)

        num_samples = len(self.signal)
        self.filtered_signal = np.zeros((num_samples, 1))

        self.filtered_signal[0] = c0 * self.signal[0]
        self.filtered_signal[1] = c0 * self.signal[1] + c1 * self.signal[0] - d1 * self.filtered_signal[0]

        for i in range(2, num_samples):
            self.filtered_signal[i] = c0 * self.signal[i-2] + c1 * self.signal[i-1] + c2 * self.signal[i] - d1 * self.filtered_signal[i-1] - d2 * self.filtered_signal[i-2]


class FourierFilter:
    def __init__(self, sample_rate):
        '''
        Instancia um filtro de fourier contendo os filtros cosseno e seno.
        Args:
            sample_rate: a quantidade de amostras de sinal por ciclo do sinal.
        '''
        self.sample_rate = sample_rate
        self.create_filter()

    def create_filter(self):
        '''
        Cria os filtros cosseno e seno de Fourier de período completo
        '''
        # Gera N argumentos espaçados entre (2.k.pi/N) e 2.pi
        filter_args = np.linspace((2*np.pi / self.sample_rate), 2*np.pi, self.sample_rate)
        self.cosine_filter = ((2 / self.sample_rate) * np.cos(filter_args))
        self.sine_filter = ((2 / self.sample_rate) * np.sin(filter_args))


class MimicFilter:
    def __init__(self, signal, tau, sample_period):
        '''
        Instancia um filtro mímico cujo objetivo é a remoção da componente CC do sinal.
        Args:
            signal: Uma sequência de amostras que compõe o sinal.
            tau: Constante de tempo da rede.
            sample_period: Período de amostragem do sinal.
        '''
        self.signal = signal
        self.tau = tau
        self.sample_period = sample_period

    def apply_filter(self):
        '''
        Aplica o filtro mímico ao sinal de entrada via
        xout(n) = k * [(1 + tau) * xin(n) - tau * xin(n-1)]
        '''
        num_samples = len(self.signal)
        self.k = 1 / np.sqrt((1 + self.tau - self.tau * np.cos(120*np.pi * self.sample_period))**2 +
                             (self.tau * np.sin(120*np.pi * self.sample_period))**2)

        self.filtered_signal = np.zeros((num_samples, 1))

        self.filtered_signal[0] = self.k * (1 + self.tau) * self.signal[0]
        for i in range(1, num_samples):
            self.filtered_signal[i] = self.k * ((1 + self.tau) * self.signal[i] - self.tau * self.signal[i-1])
