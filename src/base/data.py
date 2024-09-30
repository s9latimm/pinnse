from __future__ import annotations

import typing as t

import numpy as np

EPS = 1e-6


def arrange(start, stop, step):
    start, stop, step = int(start / EPS), int(stop / EPS), int(step / EPS)
    r = []
    while start <= stop + step / 2:
        r.append(start * EPS)
        start += step
    return np.array(r)


class Coordinate:

    @staticmethod
    def equal(a, b):
        return Coordinate(*a) == Coordinate(*b)

    def __init__(self, x: float, y: float) -> None:
        self.__x = x
        self.__y = y

    def __eq__(self, other) -> bool:
        return int(self.x / EPS) == int(other.x / EPS) and int(self.y / EPS) == int(other.y / EPS)

    def __getitem__(self, key) -> float:
        return [self.x, self.y][key]

    def __hash__(self) -> int:
        return hash((int(self.x / EPS), int(self.y / EPS)))

    def __iter__(self) -> t.Iterator[float]:
        return iter((self.x, self.y))

    def __repr__(self) -> str:
        return f'Coordinate(x={str(self.x)}, y={str(self.y)})'

    def __str__(self) -> str:
        return self.__repr__()

    def numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y


class Grid:

    @staticmethod
    def axis(start, stop, num, padding=False, extra=()) -> np.ndarray:
        p = (stop - start) / num
        if padding:
            s = np.linspace(start, stop, num, endpoint=False)
        else:
            s = np.linspace(start, stop, num, endpoint=False)
        return np.sort(np.concatenate((extra, s)))

    def __init__(self, xi: t.Sequence[int | float] = (), yi: t.Sequence[int | float] = ()) -> None:
        self.__shape = len(xi), len(yi)
        self.__grid = np.zeros(self.__shape, dtype="object")

        for i, x in enumerate(xi):
            for j, y in enumerate(yi):
                self.__grid[i][j] = Coordinate(x, y)

    def __getattr__(self, item) -> t.Any:
        return self.transform(lambda i: i.__getattribute__(item))

    def __getitem__(self, key) -> t.Any:
        if len(key) == 2:
            return self.__grid[key].flatten()
        x, y, z, *_ = key
        return self.transform(lambda i: i[z])[x, y]

    def __iter__(self) -> t.Iterator:
        return iter(self.__grid.flatten())

    def __len__(self) -> int:
        return self.__grid.size

    def __repr__(self) -> str:
        return f'{self.__grid}'

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def height(self) -> int:
        return self.__shape[1]

    @property
    def shape(self) -> t.Tuple[int, int]:
        return self.__shape

    @property
    def width(self) -> int:
        return self.__shape[0]

    def flatten(self) -> t.List[Coordinate]:
        return list(self.__grid.flatten())

    def numpy(self) -> np.ndarray:
        return self.__grid.copy()

    def transform(self, f) -> Grid:
        copy = Grid()
        copy.__grid = np.array(list(map(lambda i: np.array(list(map(f, i))), self.__grid.copy())))
        return copy
