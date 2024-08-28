import numpy as np


class AntiAliasingFilter():
    def __init__(self, period, signal, b, c):
        '''
        Instancia um filtro de anti-aliasing.
        Attributes:
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
