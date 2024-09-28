from __future__ import annotations

import typing as tp

import numpy as np

from src.base.data import Coordinate, Grid


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


class NSECloud:

    def __init__(self, grid: Grid = None) -> None:
        self.__cloud: tp.Dict[Coordinate, NSEFact] = dict()
        if grid is not None:
            for k in grid:
                self.__cloud[k] = NSEFact()

    def __contains__(self, key):
        return key in self.__cloud.keys()

    def __getitem__(self, key: tp.Tuple | Coordinate) -> NSEFact:
        k = Coordinate(*key)
        if k not in self.__cloud:
            raise KeyError(k)
        return self.__cloud[k]

    def __iter__(self):
        return iter(self.__cloud.items())

    def __len__(self):
        return len(self.__cloud)

    def __repr__(self) -> str:
        return f'{self.__cloud}'

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def keys(self) -> tp.Set[Coordinate]:
        return set(self.__cloud.keys())

    def add(self, key: tp.Tuple | Coordinate, **kwargs):
        k = Coordinate(*key)
        if k in self.__cloud:
            raise KeyError(k)
        self.__cloud[k] = NSEFact(**kwargs)
        return self.__cloud[k]

    def copy(self) -> NSECloud:
        cloud = NSECloud()
        cloud.__cloud = dict(self.__cloud)
        return cloud

    def detach(self):
        return list(self.__cloud.copy().items())

    def numpy(self) -> np.ndarray:
        return np.array([np.concatenate([k.numpy(), v.numpy()]) for k, v in self.__cloud.items()])
