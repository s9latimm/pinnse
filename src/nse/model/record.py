from __future__ import annotations

import typing as tp

import numpy as np


class Record:

    def __init__(self, u: float = np.nan, v: float = np.nan, p: float = np.nan) -> None:
        self.__u = u
        self.__v = v
        self.__p = p

    def __iter__(self) -> tp.Iterator[float]:
        return iter((self.u, self.v, self.p))

    @property
    def labels(self) -> list[str]:
        return ['u', 'v', 'p']

    def __repr__(self) -> str:
        return f'Fact(u={str(self.u)}, v={str(self.v)}, p={str(self.p)})'

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def p(self) -> float:
        return self.__p

    @property
    def u(self) -> float:
        return self.__u

    @property
    def v(self) -> float:
        return self.__v

    def __sub__(self, other: Record) -> Record:
        return Record(abs(self.u - other.u), abs(self.v - other.v), abs(self.p - other.p))
