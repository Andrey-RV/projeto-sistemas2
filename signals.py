import numpy as np
import numpy.typing as npt
from typing import Generator
from dataclasses import dataclass


@dataclass
class Signals:
    """Armazena um conjunto de sinais e tensões trifásicas, além do vetor de tempo e o período de amostragem.

    Yields:
        va (npt.NDArray[np.float64]): Tensão fase A.
        vb (npt.NDArray[np.float64]): Tensão fase B.
        vc (npt.NDArray[np.float64]): Tensão fase C.
        ia (npt.NDArray[np.float64]): Corrente fase A.
        ib (npt.NDArray[np.float64]): Corrente fase B.
        ic (npt.NDArray[np.float64]): Corrente fase C.
        t (npt.NDArray[np.float64]): Vetor de tempo.
        sampling_period (float): Período de amostragem.
    """
    va: npt.NDArray[np.float64]
    vb: npt.NDArray[np.float64]
    vc: npt.NDArray[np.float64]
    ia: npt.NDArray[np.float64]
    ib: npt.NDArray[np.float64]
    ic: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    sampling_period: float

    def __post_init__(self) -> None:
        """Certifica-se de que todos os atributos são do tipo np.ndarray.

        Returns:
            None
        """
        self.va = np.array(self.va)
        self.vb = np.array(self.vb)
        self.vc = np.array(self.vc)
        self.ia = np.array(self.ia)
        self.ib = np.array(self.ib)
        self.ic = np.array(self.ic)
        self.t = np.array(self.t)

    def __iter__(self) -> Generator[tuple[str, npt.NDArray[np.float64]], None, None]:
        """Itera os atributos do objeto em forma de tuplas ('nome', valor).

        Yields:
            Tuple[str, npt.NDArray[np.float64]]: Tuplas contendo o nome do atributo e seu valor.

        Returns:
            Generator[tuple[str, npt.NDArray[np.float64]], None, None]
        """
        yield 'va', self.va
        yield 'vb', self.vb
        yield 'vc', self.vc
        yield 'ia', self.ia
        yield 'ib', self.ib
        yield 'ic', self.ic

    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)
