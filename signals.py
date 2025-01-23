import numpy as np
import numpy.typing as npt
from typing import Union, Generator
from pandas import Series  # type: ignore
from dataclasses import dataclass, field


numeric_array = Union[npt.NDArray[np.float64 | np.complex128], Series]
name_value_pair = tuple[str, numeric_array]


@dataclass
class Signals:
    """Armazena um conjunto de sinais e tensões trifásicas, além do vetor de tempo e o período de amostragem.

    Yields:
        va (numeric_array): Tensão fase A.
        vb (numeric_array): Tensão fase B.
        vc (numeric_array): Tensão fase C.
        ia (numeric_array): Corrente fase A.
        ib (numeric_array): Corrente fase B.
        ic (numeric_array): Corrente fase C.
        t (numeric_array): Vetor de tempo.
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

    def get_voltages(self) -> tuple[name_value_pair, name_value_pair, name_value_pair]:
        """Retorna as tensões trifásicas do objeto.

        Returns:
            npt.NDArray[np.float64 | np.complex128]: As tensões trifásicas.
        """
        return ('va', self.va), ('vb', self.vb), ('vc', self.vc)

    def get_currents(self) -> tuple[name_value_pair, name_value_pair, name_value_pair]:
        """Retorna as correntes trifásicas do objeto.

        Returns:
            npt.NDArray[np.float64 | np.complex128]: As correntes trifásicas.
        """
        return ('ia', self.ia), ('ib', self.ib), ('ic', self.ic)
