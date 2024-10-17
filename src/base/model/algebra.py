from __future__ import annotations

from pathlib import Path


class RealNumber:
    EPS: float = 1e-3
    DELTA: float = 1e-5

    def __init__(self, value: float) -> None:
        if value < 0:
            self.__value: int = int((value - self.DELTA) / self.EPS)
        else:
            self.__value: int = int((value + self.DELTA) / self.EPS)

    def __eq__(self, other: RealNumber | float) -> bool:
        # pylint: disable=protected-access
        return self.__value == RealNumber(float(other)).__value

    def __gt__(self, other: RealNumber | float) -> bool:
        # pylint: disable=protected-access
        return self.__value > RealNumber(float(other)).__value

    def __ge__(self, other: RealNumber | float) -> bool:
        # pylint: disable=protected-access
        return self.__value >= RealNumber(float(other)).__value

    def __lt__(self, other: RealNumber | float) -> bool:
        # pylint: disable=protected-access
        return self.__value <= RealNumber(float(other)).__value

    def __le__(self, other: RealNumber | float) -> bool:
        # pylint: disable=protected-access
        return self.__value <= RealNumber(float(other)).__value

    def __add__(self, other: RealNumber | float) -> RealNumber:
        return RealNumber(float(self) + float(other))

    def __radd__(self, other) -> RealNumber:
        return self.__add__(other)

    def __sub__(self, other: RealNumber | float) -> RealNumber:
        return RealNumber(float(self) - float(other))

    def __rsub__(self, other) -> RealNumber:
        return RealNumber(float(other) - float(self))

    def __mul__(self, other: RealNumber | float) -> RealNumber:
        return RealNumber(float(self) * float(other))

    def __rmul__(self, other: RealNumber | float) -> RealNumber:
        return self.__mul__(other)

    def __truediv__(self, other: RealNumber | float) -> RealNumber:
        return RealNumber(float(self) / float(other))

    def __rtruediv__(self, other: RealNumber | float) -> RealNumber:
        return RealNumber(float(other) / float(self))

    def __float__(self) -> float:
        return self.__value * self.EPS

    def __hash__(self) -> int:
        return self.__value

    def __str__(self):
        s = f'{self.__value:04d}'
        return f'{s[:-3]}_{s[-3:]}'

    def __repr__(self):
        return self.__str__()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(f'{str(self)}\n')


class Integer:

    def __init__(self, value: int) -> None:
        self.__value: int = value

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(f'{self.__value:d}\n')
