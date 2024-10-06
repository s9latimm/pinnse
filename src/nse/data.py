from __future__ import annotations

import typing as tp

import numpy as np

from src.base.geometry import Coordinate, Cloud


class NSEFact:

    def __init__(self, u: float = np.nan, v: float = np.nan, p: float = np.nan) -> None:
        self.__u = u
        self.__v = v
        self.__p = p

    def __getitem__(self, key) -> float:
        return [self.u, self.v, self.p][key]

    def __iter__(self) -> tp.Iterator[float]:
        return iter((self.u, self.v, self.p))

    def __repr__(self) -> str:
        return f'Fact(u={str(self.u)}, v={str(self.v)}, p={str(self.p)})'

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def p(self) -> float:
        return self.__p

    @p.setter
    def p(self, value: float) -> None:
        self.__p = value

    @property
    def u(self) -> float:
        return self.__u

    @u.setter
    def u(self, value: float) -> None:
        self.__u = value

    @property
    def v(self) -> float:
        return self.__v

    @v.setter
    def v(self, value: float) -> None:
        self.__v = value

    def numpy(self) -> np.ndarray:
        return np.array([self.__u, self.__v, self.p])

    def update(self, u: float = None, v: float = None, p: float = None) -> None:
        if u is not None:
            self.__u = u
        if v is not None:
            self.__v = v
        if p is not None:
            self.__p = p


class NSECloud(Cloud):

    def __getitem__(self, key: tuple | Coordinate) -> NSEFact:
        return super().__getitem__(key)

    def add(self, key: tuple | Coordinate, **kwargs) -> NSEFact:
        return super().add(key, NSEFact(**kwargs))

    def copy(self) -> NSECloud:
        return super().copy()

    def detach(self) -> tp.List[tuple[Coordinate, NSEFact]]:
        return super().detach()
