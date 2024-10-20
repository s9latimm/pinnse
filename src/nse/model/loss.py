from __future__ import annotations

import logging
import typing as tp
from pathlib import Path


class Loss:

    def __init__(self, n: float, f: float, g: float, u: float, v: float, loss: float) -> None:
        self.__n = int(n)
        self.__f = float(f)
        self.__g = float(g)
        self.__u = float(u)
        self.__v = float(v)
        self.__loss = float(loss)

    def __iter__(self) -> tp.Iterator[float]:
        return iter((self.__n, self.__f, self.__g, self.__u, self.__v, self.__loss))

    def __str__(self) -> str:
        return f'Loss(n={self.__n}, f={self.__f}, g={self.__g}, u={self.__u}, v={self.__v}, loss={self.__loss})'

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def n(self) -> float:
        return self.__n

    @property
    def f(self) -> float:
        return self.__f

    @property
    def g(self) -> float:
        return self.__g

    @property
    def u(self) -> float:
        return self.__u

    @property
    def v(self) -> float:
        return self.__v

    def __float__(self) -> float:
        return self.__loss

    # def save(self, path: Path):
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     with path.open("w", encoding="utf-8") as f:
    #         f.write(f'{self.u:.16f},{self.v:.16f},{self.p:.16f}\n')

    # @staticmethod
    # def load(path: Path) -> Loss:
    #     if path.exists():
    #         u, v, p = path.read_text(encoding='utf-8').strip().split(',')
    #         return Loss(float(u), float(v), float(p))
    #     else:
    #         logging.error(f'{path} does not exist')


class Losses:

    def __init__(self, losses: list[Loss] = ()) -> None:
        self.__losses = list(losses)

    def __iter__(self) -> tp.Iterator[Loss]:
        return iter(self.__losses)

    @staticmethod
    def load(path: Path) -> Losses:
        # pylint: disable=protected-access
        if path.exists():
            loss = Losses()
            lines = path.read_text(encoding='utf-8').strip().splitlines()
            for line in lines:
                loss.__losses.append(Loss(*line.strip().split(',')))
            return loss
        logging.error(f'{path} does not exist')
        raise FileNotFoundError
