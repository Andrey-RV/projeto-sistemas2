import numpy as np
import numpy.typing as npt


class AntiAliasingFilter:
    def __init__(self, signal: npt.NDArray[np.float64], sampling_period: float, b: float, c: float) -> None:
        '''Instancia um filtro de anti-aliasing.
        Args:
            signal (npt.NDArray[np.float64]): O sinal a ser filtrado.
            sampling_period (float): O período de amostragem do sinal.
            b (float): Coeficiente b do filtro.
            c (float): Coeficiente c do filtro.

        Returns:
            None
        '''
        self.__signal = signal
        self.__sampling_period = sampling_period
        self.__a = 2 / self.__sampling_period
        self.__b = b
        self.__c = c

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
            f"AntiAliasingFilter("
            f"{signal_repr}, "
            f"sampling_period={self.__sampling_period}, "
            f"a={self.__a}, "
            f"b={self.__b}, "
            f"c={self.__c}, "
            f"filtered_signal={'set' if hasattr(self, 'filtered_signal') else 'unset'})"
        )

    def apply_filter(self) -> None:
        r'''Aplica o filtro anti-aliasing ao sinal de entrada

            $$x_0 = c_0.x_0$$
            $$x_1 = c_0.x_1 + c_1.x_0 - d_1.x_0$$
            $$ \x_{out}(n) = \left[c_0.x_{in}(n) + c_1.x_{in}(n-1) + c_2.x_{in}(n-2)\right] - \left[d_1.x_{out}(n-1) + d_2.x_{out}(n-2)right], \quad n \geq 2$$

            Returns:
                None
        '''
        c0 = self.__c / (self.__a**2 + self.__a*self.__b + self.__c)
        c1 = 2 * c0
        c2 = c0
        d1 = 2 * (self.__c - self.__a**2) / (self.__a**2 + self.__a*self.__b + self.__c)
        d2 = (self.__a**2 + self.__c - self.__a*self.__b) / (self.__a**2 + self.__a*self.__b + self.__c)

        self.filtered_signal = np.zeros((len(self.__signal)))
        self.filtered_signal[0] = c0 * self.__signal[0]
        self.filtered_signal[1] = c0 * self.__signal[1] + c1 * self.__signal[0] - d1 * self.filtered_signal[0]

        for i in range(2, len(self.__signal)):
            self.filtered_signal[i] = c0 * self.__signal[i-2] + c1 * self.__signal[i-1] + c2 * self.__signal[i] \
                                      - d1 * self.filtered_signal[i-1] - d2 * self.filtered_signal[i-2]


class FourierFilter:
    def __init__(self, samples_per_cycle: int, mode: str = 'full') -> None:
        '''Instancia um filtro de fourier contendo os filtros cosseno e seno.
        Args:
            samples_per_cycle (int): A quantidade de amostras capturadas pelo IED em um período da onda fundamental.

        Returns:
            None
        '''
        self.__samples_per_cycle = samples_per_cycle
        self.__mode = mode

    def __repr__(self) -> str:
        return (
            f"FourierFilter("
            f"mode={self.__mode}, "
            f"samples_per_cycle={self.__samples_per_cycle}, "
            f"cosine_filter={'set' if hasattr(self, 'cosine_filter') else 'unset'}, "
            f"sine_filter={'set' if hasattr(self, 'sine_filter') else 'unset'})"
        )

    def create_filter(self) -> None:
        r'''Cria os filtros cosseno e seno de Fourier para a quantidade de amostras por ciclo especificada.
              $$\cos(k) = \frac{2}{N} \sum_{k=1}^{N} \cos(2\pi k/N)$$
              $$\sin(k) = \frac{2}{N} \sum_{k=1}^{N} \sin(2\pi k/N)$$

        Returns:
            None
        '''
        if self.__mode == 'full':
            filter_args = np.linspace(2*np.pi / self.__samples_per_cycle, 2*np.pi, self.__samples_per_cycle)
            self.cosine_filter = (2 / self.__samples_per_cycle * np.cos(filter_args))
            self.sine_filter = (2 / self.__samples_per_cycle * np.sin(filter_args))
        else:  # TODO: Implementar o modo 'half'
            raise NotImplementedError('O modo half ainda não foi implementado.')


class MimicFilter:
    def __init__(self, signal: npt.NDArray[np.float64], tau: float, sampling_period: float) -> None:
        '''Instancia um filtro mímico cujo objetivo é a remoção da componente CC do sinal.
        Args:
            signal (npt.NDArray[np.float64]): O sinal a ser filtrado.
            tau (float): O valor de tau do sistema.
            sampling_period (float): O período de amostragem do sinal.

        Returns:
            None
        '''
        self.__signal = signal
        self.__tau = tau
        self.__sampling_period = sampling_period

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
            f"MimicFilter("
            f"{signal_repr}, "
            f"tau={self.__tau}, "
            f"sampling_period={self.__sampling_period}, "
            f"k={'set' if hasattr(self, 'k') else 'unset'}, "
            f"filtered_signal={'set' if hasattr(self, 'filtered_signal') else 'unset'})"
        )

    def apply_filter(self) -> None:
        r'''
        Aplica o filtro mímico ao sinal de entrada
        $$k = \frac{1}{\sqrt{(1 + \tau - \tau \cos(120\pi T_s))^2 + (\tau \sin(120\pi T_s))^2}}$$
        $$x_0 = k(1 + \tau)x_0$$
        $$x_{out}(n) = k \left[(1 + \tau) x_{in}(n) - \tau x_{in}(n-1)\right], \quad n \geq 1$$

        Returns:
            None
        '''
        self.filtered_signal = np.zeros(len(self.__signal))

        self.k = 1 / np.sqrt(
                (1 + self.__tau - self.__tau * np.cos(120*np.pi * self.__sampling_period))**2 +
                (self.__tau * np.sin(120*np.pi * self.__sampling_period))**2
            )

        self.filtered_signal[0] = self.k * (1 + self.__tau) * self.__signal[0]
        for i in range(1, len(self.__signal)):
            self.filtered_signal[i] = self.k * ((1 + self.__tau) * self.__signal[i] - self.__tau * self.__signal[i-1])
