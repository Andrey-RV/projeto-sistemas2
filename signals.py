import numpy as np
import numpy.typing as npt
from typing import Generator, TypeAlias
from dataclasses import dataclass, field, fields

name_value_pair: TypeAlias = tuple[str, npt.NDArray[np.complex128]]


@dataclass
class Signals:
    """Armazena um conjunto de sinais e tensões trifásicas, além do vetor de tempo e o período de amostragem.

    Yields:
        va (npt.NDArray[np.complex128]): Tensão fase A.
        vb (npt.NDArray[np.complex128]): Tensão fase B.
        vc (npt.NDArray[np.complex128]): Tensão fase C.
        ia (npt.NDArray[np.complex128]): Corrente fase A.
        ib (npt.NDArray[np.complex128]): Corrente fase B.
        ic (npt.NDArray[np.complex128]): Corrente fase C.
        t (npt.NDArray[np.float64]): Vetor de tempo.
        sampling_period (float): Período de amostragem.
    """
    va: npt.NDArray[np.complex128] = field(default_factory=lambda: np.array([], dtype=np.complex128))
    vb: npt.NDArray[np.complex128] = field(default_factory=lambda: np.array([], dtype=np.complex128))
    vc: npt.NDArray[np.complex128] = field(default_factory=lambda: np.array([], dtype=np.complex128))
    ia: npt.NDArray[np.complex128] = field(default_factory=lambda: np.array([], dtype=np.complex128))
    ib: npt.NDArray[np.complex128] = field(default_factory=lambda: np.array([], dtype=np.complex128))
    ic: npt.NDArray[np.complex128] = field(default_factory=lambda: np.array([], dtype=np.complex128))
    t: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    sampling_period: float = field(default=np.nan)

    def __post_init__(self) -> None:
        """Certifica-se de que todos os sinais estão armazenados como arrays numpy."""
        for field_ in fields(self):
            if field_.name != 'sampling_period':
                setattr(self, field_.name, np.asarray(getattr(self, field_.name)))

    def __iter__(self) -> Generator[tuple[str, npt.NDArray[np.float64]], None, None]:
        """Itera os atributos do objeto em forma de tuplas ('nome', valor).

        Returns:
            Generator[tuple[str, npt.NDArray[np.float64 | np.complex128]], None, None]
        """
        signals = [field_.name for field_ in fields(self) if field_.name not in ('t', 'sampling_period')]
        for signal in signals:
            yield signal, getattr(self, signal)

    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)

    def __getitem__(self, key) -> npt.NDArray[np.float64]:
        return getattr(self, key)

    def get_voltages(self) -> tuple[name_value_pair, name_value_pair, name_value_pair]:
        """Retorna as tensões trifásicas do objeto.

        Returns:
            npt.NDArray[np.float64 | np.complex128]: Pares (name, value) das tensões trifásicas.
        """
        return ('va', self.va), ('vb', self.vb), ('vc', self.vc)

    def get_currents(self) -> tuple[name_value_pair, name_value_pair, name_value_pair]:
        """Retorna as correntes trifásicas do objeto.

        Returns:
            npt.NDArray[np.float64 | np.complex128]: Pares (name, value) das correntes trifásicas.
        """
        return ('ia', self.ia), ('ib', self.ib), ('ic', self.ic)
