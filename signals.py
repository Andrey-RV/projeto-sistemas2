import numpy as np
import numpy.typing as npt
from typing import Generator
from dataclasses import dataclass, field


numeric_array = npt.NDArray[np.float64 | np.complex128]


@dataclass
class Signals:
    """Armazena um conjunto de sinais e tensões trifásicas, além do vetor de tempo e o período de amostragem.

    Yields:
        va (npt.NDArray[np.float64 | np.complex128]): Tensão fase A.
        vb (npt.NDArray[np.float64 | np.complex128]): Tensão fase B.
        vc (npt.NDArray[np.float64 | np.complex128]): Tensão fase C.
        ia (npt.NDArray[np.float64 | np.complex128]): Corrente fase A.
        ib (npt.NDArray[np.float64 | np.complex128]): Corrente fase B.
        ic (npt.NDArray[np.float64 | np.complex128]): Corrente fase C.
        t (npt.NDArray[np.float64 | np.complex128]): Vetor de tempo.
        sampling_period (float): Período de amostragem.
    """
    va: numeric_array = field(default_factory=lambda: np.array([], dtype=np.float64))
    vb: numeric_array = field(default_factory=lambda: np.array([], dtype=np.float64))
    vc: numeric_array = field(default_factory=lambda: np.array([], dtype=np.float64))
    ia: numeric_array = field(default_factory=lambda: np.array([], dtype=np.float64))
    ib: numeric_array = field(default_factory=lambda: np.array([], dtype=np.float64))
    ic: numeric_array = field(default_factory=lambda: np.array([], dtype=np.float64))
    t: numeric_array = field(default_factory=lambda: np.array([], dtype=np.float64))
    sampling_period: float = field(default=np.nan)

    def __post_init__(self) -> None:
        """Certifica-se de que todos os sinais estão armazenados como arrays numpy."""
        for name in ["va", "vb", "vc", "ia", "ib", "ic"]:
            setattr(self, name, np.asarray(getattr(self, name)))

        self.t = np.asarray(self.t)

    def __iter__(self) -> Generator[tuple[str, numeric_array], None, None]:
        """Itera os atributos do objeto em forma de tuplas ('nome', valor).

        Returns:
            Generator[tuple[str, npt.NDArray[np.float64 | np.complex128]], None, None]
        """
        for name in ["va", "vb", "vc", "ia", "ib", "ic"]:
            yield name, getattr(self, name)

    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)

    def __getitem__(self, key) -> numeric_array:
        return getattr(self, key)

    def get_voltage(self) -> tuple[numeric_array, numeric_array, numeric_array]:
        """Retorna as tensões trifásicas do objeto.

        Returns:
            npt.NDArray[np.float64 | np.complex128]: As tensões trifásicas.
        """
        return self.va, self.vb, self.vc

    def get_current(self) -> tuple[numeric_array, numeric_array, numeric_array]:
        """Retorna as correntes trifásicas do objeto.

        Returns:
            npt.NDArray[np.float64 | np.complex128]: As correntes trifásicas.
        """
        return self.ia, self.ib, self.ic
