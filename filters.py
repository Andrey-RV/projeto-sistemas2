from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.signal import sosfilt


class AntiAliasingFilter:
    def __init__(self, sampling_period: float, b: float, c: float) -> None:
        '''Implementa um filtro de anti-aliasing usando um filtro IIR de segunda ordem.

        Args:
            sampling_period (float): O período de amostragem do sinal.
            b (float): Coeficiente b do filtro.
            c (float): Coeficiente c do filtro.
        '''
        self._sampling_period = sampling_period
        self._a = 2 / self._sampling_period
        self._b = b
        self._c = c
        self._init_coeffs()

    def _init_coeffs(self) -> None:
        """Pré calcula os coeficientes do filtro."""
        denominator = self._a**2 + self._a*self._b + self._c
        c0 = self._c / denominator
        c1 = 2*self._c / denominator
        c2 = self._c / denominator
        d1 = 2*(self._c - self._a**2) / denominator
        d2 = (self._a**2 + self._c - self._a*self._b) / denominator

        self._sos = np.array([[c0, c1, c2, 1, d1, d2]])

    def apply_filter(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.floating]:
        r'''Aplica um filtro de anti-aliasing ao sinal de entrada.

            Essa operação pode ser descrita pela equação de diferenças:
            $$ \x_{out}(n)=\left[c_0.x_{in}(n)+c_1.x_{in}(n-1)+c_2.x_{in}(n-2)\right]-
            \left[d_1.x_{out}(n-1)+d_2.x_{out}(n-2)\right]$$

            ou, no domínio Z:
            $$H(z)=\frac{c_0+c_1z^{-1}+c_2z^{-2}}{1+d_1z^{-1}+d_2z^{-2}}$$

            Returns:
                npt.NDArray[np.floating]: O sinal filtrado.
        '''
        return sosfilt(self._sos, signal)


class FourierFilter:
    def __init__(self, samples_per_cycle: int, mode: str = "full") -> None:
        '''Instancia um filtro de fourier contendo os filtros cosseno e seno.

        Args:
            samples_per_cycle (int): A quantidade de amostras capturadas pelo IED em um período da onda fundamental.
            mode (str): O modo de operação do filtro. Pode ser "full" ou "half". Default: "full".
        '''
        self._samples_per_cycle = samples_per_cycle
        self._mode = mode

    def create_filter(self) -> None:
        r'''Cria os filtros cosseno e seno de Fourier para a quantidade de amostras por ciclo especificada.
          Para cada amostra $n$ com $n = 0, 1, \ldots, N-1$, os filtros são definidos como:

          $$c[n] = \frac{2}{N} \cos\left(\frac{2\pi n}{N}\right)$$
          $$s[n] = \frac{2}{N} \sin\left(\frac{2\pi n}{N}\right)$$

          onde $N$ é o número total de amostras por ciclo.
          '''
        if self._mode == "full":
            args = np.linspace(0, 2*np.pi, self._samples_per_cycle, endpoint=False)
            self.cosine_filter = (2 / self._samples_per_cycle * np.cos(args))
            self.sine_filter = (2 / self._samples_per_cycle * np.sin(args))
        else:  # TODO: Implementar o modo "half"
            raise NotImplementedError("O modo 'half' ainda não foi implementado.")


class MimicFilter:
    def __init__(self, tau: float, sampling_period: float, freq: float = 60.0) -> None:
        '''Instancia um filtro mímico cujo objetivo é a remoção da componente CC do sinal.
        Args:

            tau (float): O valor de tau do sistema.
            sampling_period (float): O período de amostragem do sinal.
            freq (float): A frequência da onda fundamental. Default: 60.0.
        '''
        self._tau = tau
        self._sampling_period = sampling_period
        self._freq = freq
        self._init_coefficients()

    def _init_coefficients(self) -> None:
        """Pre-calculates filter coefficients"""
        arg = 2 * np.pi * self._freq * self._sampling_period
        cos_term = np.cos(arg)
        sin_term = np.sin(arg)
        denominator = np.sqrt((1 + self._tau - self._tau * cos_term)**2 + (self._tau * sin_term)**2)

        self._k = 1 / denominator
        self._sos = np.array([[self._k * (1 + self._tau), -self._k * self._tau, 0, 1, 0, 0]])

    def apply_filter(self, signal: npt.NDArray[np.float64]) -> npt.NDArray[np.floating]:
        r'''Retorna um sinal filtrado pelo filtro mímico.

            Essa operação pode ser descrita pela equação de diferenças:
            $$x_{out}(n) = k \left[(1 + \tau) x_{in}(n) - \tau x_{in}(n-1)\right]$$

            ou, no domínio Z:
            $$H(z) = \frac{k(1 + \tau) - k\tau z^{-1}}{1}$$
        '''
        return sosfilt(self._sos, signal)
