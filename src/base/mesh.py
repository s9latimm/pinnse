from __future__ import annotations

import typing as tp
from abc import abstractmethod

import numpy as np

EPS: float = 1e-6
DELTA: float = 1e-7


def clamp(f: float) -> int:
    return int((f + DELTA) / EPS)


def eps(i: int) -> float:
    return i * EPS


def merge(*lists: tp.Sequence[tp.Any]) -> list[tp.Any]:
    merged = []
    for xs in lists:
        for x in xs:
            if x not in merged:
                merged.append(x)
    return merged


def equal(a: float, b: float) -> bool:
    return clamp(a) == clamp(b)


def leq(a: float, b: float) -> bool:
    return clamp(a) <= clamp(b)


def arrange(start: float, stop: float, step: float) -> list[float]:
    start, stop, step = clamp(start), clamp(stop), clamp(step)
    r = []
    if step <= 0.:
        return r
    if start < stop:
        while start <= stop:
            r.append(eps(start))
            start += step
    elif start > stop:
        while start >= stop:
            r.append(eps(start))
            start -= step
    else:
        return [eps(start)]
    return r


class Coordinate:

    def __init__(self, x: float, y: float) -> None:
        self.__x = x
        self.__y = y

    def __eq__(self, other) -> bool:
        return clamp(self.x) == clamp(other.x) and clamp(self.y) == clamp(other.y)

    def __getitem__(self, key) -> float:
        return [self.x, self.y][key]

    def __hash__(self) -> int:
        return hash((clamp(self.x), clamp(self.y)))

    def __iter__(self) -> tp.Iterator[float]:
        return iter((self.x, self.y))

    def __repr__(self) -> str:
        return f'Coordinate(x={str(self.x)}, y={str(self.y)})'

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, coordinate: tuple[float, float] | Coordinate) -> Coordinate:
        c = Coordinate(*coordinate)
        return Coordinate(self.x + c.x, self.y + c.y)

    def __sub__(self, coordinate: tuple[float, float] | Coordinate) -> Coordinate:
        c = Coordinate(*coordinate)
        return Coordinate(self.x - c.x, self.y - c.y)

    def __mul__(self, factor: float) -> Coordinate:
        return Coordinate(self.x * factor, self.y * factor)

    def __truediv__(self, factor: float) -> Coordinate:
        if equal(factor, 0):
            return Coordinate(np.infty, np.infty)
        return Coordinate(self.x / factor, self.y / factor)

    def distance(self, coordinate: tuple[float, float] | Coordinate) -> float:
        c = Coordinate(*coordinate)
        return np.sqrt((self.__x - c.x)**2 + (self.__y - c.y)**2)

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    def rotate(self, angle: float) -> Coordinate:
        angle = angle * np.pi / 180
        return Coordinate(
            self.__x * np.cos(angle) - self.__y * np.sin(angle),
            self.__x * np.sin(angle) + self.__y * np.cos(angle),
        )


class Axis:

    def __init__(self, label: str, start: float, stop: float) -> None:
        self.__start = start
        self.__stop = stop
        self.__label = label

    @property
    def dim(self) -> tuple[float, float]:
        return self.__start, self.__stop

    @property
    def start(self) -> float:
        return self.__start

    @property
    def stop(self) -> float:
        return self.__stop

    @property
    def shape(self) -> tuple[float, float]:
        return self.__start, self.__stop

    @property
    def label(self) -> str:
        return self.__label

    def arrange(self, step: float, center=False, padding=0) -> list[float]:
        assert step > 0
        c = step / 2 if center else 0
        return arrange(self.__start + c - padding, self.__stop - c + padding, step)


class Grid:

    def __init__(self, xs: tp.Sequence[float], ys: tp.Sequence[float]) -> None:
        self.__width = len(xs)
        self.__height = len(ys)

        self.__grid = np.zeros((self.__width, self.__height), dtype="object")
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                self.__grid[i][j] = Coordinate(x, y)

    def __getattr__(self, item) -> np.ndarray:
        # pylint: disable=protected-access
        return self.map(lambda i: i.__getattribute__(item)).__grid

    def __iter__(self) -> tp.Iterator:
        return iter(self.__grid.flatten())

    def __len__(self) -> int:
        return self.__grid.size

    def __repr__(self) -> str:
        return f'{self.__grid}'

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def shape(self) -> tuple[int, int]:
        return self.__width, self.__height

    def flatten(self) -> list[Coordinate]:
        return list(self.__grid.flatten())

    def transform(self, mesh: Mesh):
        return self.map(lambda i: mesh[i])

    def map(self, f) -> Grid:
        # pylint: disable=protected-access
        copy = Grid([], [])
        copy.__grid = np.array(list(map(lambda i: np.array(list(map(f, i))), self.__grid.copy())))
        copy.__width = self.__width
        copy.__height = self.__height
        return copy

    def mesh(self) -> Mesh:
        mesh = Mesh()
        for c in self:
            mesh.insert(c)
        return mesh


class Mesh:

    def __init__(self) -> None:
        self.__mesh: dict[Coordinate, tp.Any] = {}

    def __contains__(self, coordinate: tuple[float, float] | Coordinate) -> bool:
        return Coordinate(*coordinate) in self.__mesh.keys()

    def __getitem__(self, coordinate: tuple[float, float] | Coordinate) -> tp.Any:
        c = Coordinate(*coordinate)
        if c not in self:
            raise KeyError(c)
        return self.__mesh[c]

    def grid(self) -> Grid:
        return Grid(sorted({i.x for i in self.__mesh.keys()}), sorted({i.y for i in self.__mesh.keys()}))

    def keys(self) -> list[Coordinate]:
        return list(self.__mesh.keys())

    def __iter__(self) -> tp.Iterator[tuple[Coordinate, tp.Any]]:
        return iter(self.__mesh.items())

    def __len__(self) -> int:
        return len(self.__mesh)

    def __repr__(self) -> str:
        return f'{self.__mesh}'

    def __str__(self) -> str:
        return self.__repr__()

    def insert(self, coordinate: tuple[float, float] | Coordinate, value: tp.Any | None) -> tp.Any:
        c = Coordinate(*coordinate)
        if c in self.__mesh:
            raise KeyError(c)
        self.__mesh[c] = value
        return self.__mesh[c]

    @abstractmethod
    def emplace(self, key: tuple | Coordinate, **kwargs) -> tp.Any:
        ...

    def clear(self) -> None:
        self.__mesh.clear()

    def copy(self) -> tp.Any:
        # pylint: disable=protected-access
        mesh = Mesh()
        mesh.__mesh = self.__mesh.copy()
        return mesh

    def detach(self) -> list[tuple[Coordinate, tp.Any]]:
        return list(self.__mesh.copy().items())

    def numpy(self) -> np.ndarray:
        return np.array([np.concatenate([k.numpy(), v.numpy()]) for k, v in self.__mesh.items()])
