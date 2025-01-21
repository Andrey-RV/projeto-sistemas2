import numpy as np
from dataclasses import dataclass


@dataclass
class Signals:
    va: np.ndarray
    vb: np.ndarray
    vc: np.ndarray
    ia: np.ndarray
    ib: np.ndarray
    ic: np.ndarray
    t: np.ndarray
    sampling_period: float

    def __post_init__(self):
        self.va = np.array(self.va)
        self.vb = np.array(self.vb)
        self.vc = np.array(self.vc)
        self.ia = np.array(self.ia)
        self.ib = np.array(self.ib)
        self.ic = np.array(self.ic)
        self.t = np.array(self.t)

    def __iter__(self):
        yield 'va', self.va
        yield 'vb', self.vb
        yield 'vc', self.vc
        yield 'ia', self.ia
        yield 'ib', self.ib
        yield 'ic', self.ic

    def __setitem__(self, key, value):
        setattr(self, key, value)
