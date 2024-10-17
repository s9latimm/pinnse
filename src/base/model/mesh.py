from __future__ import annotations

import typing as tp
from abc import abstractmethod
from pathlib import Path

import numpy as np

from src.base.model.algebra import RealNumber


def merge(*lists: tp.Sequence[tp.Any]) -> list[tp.Any]:
    merged = []
    for xs in lists:
        for x in xs:
            if x not in merged:
                merged.append(x)
    return merged


def arrange(start: float, stop: float, step: float, center=False) -> list[float]:
    start, stop, step = RealNumber(start), RealNumber(stop), RealNumber(step)

    if center:
        start = start + step / 2
        stop = stop - step / 2

    if not step > 0:
        return []

    r = []
    if start < stop:
        while start <= stop:
            r.append(start)
            start += step
    elif start > stop:
        while start >= stop:
            r.append(start)
            start -= step
    else:
        r.append(start)

    return [float(i) for i in r]


class Coordinate:

    def __init__(self, x: float | RealNumber, y: float | RealNumber) -> None:
        self.__x = RealNumber(float(x))
        self.__y = RealNumber(float(y))

    def __eq__(self, coordinate: tuple[float, float] | Coordinate) -> bool:
        c = Coordinate(*coordinate)
        return self.__x == c.__x and self.__y == c.__y

    def __hash__(self) -> int:
        return hash((hash(self.__x), hash(self.__y)))

    def __iter__(self) -> tp.Iterator[float]:
        return iter([float(self.__x), float(self.__y)])

    def __repr__(self) -> str:
        return f'Coordinate(x={str(self.__x)}, y={str(self.__y)})'

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, coordinate: tuple[float, float] | Coordinate) -> Coordinate:
        c = Coordinate(*coordinate)
        return Coordinate(self.__x + c.__x, self.__y + c.__y)

    def __radd__(self, coordinate: tuple[float, float] | Coordinate) -> Coordinate:
        return self.__add__(coordinate)

    def __sub__(self, coordinate: tuple[float, float] | Coordinate) -> Coordinate:
        c = Coordinate(*coordinate)
        return Coordinate(self.__x - c.__x, self.__y - c.__y)

    def __rsub__(self, coordinate: tuple[float, float] | Coordinate) -> Coordinate:
        c = Coordinate(*coordinate)
        return Coordinate(c.__x - self.__x, c.__y - self.__y)

    def __mul__(self, factor: float) -> Coordinate:
        return Coordinate(self.__x * factor, self.__y * factor)

    def __rmul__(self, factor: float) -> Coordinate:
        return self.__mul__(factor)

    def __truediv__(self, factor: float) -> Coordinate:
        if RealNumber(factor) == 0:
            return Coordinate(np.infty, np.infty)
        return Coordinate(self.__x / factor, self.__y / factor)

    def distance(self, coordinate: tuple[float, float] | Coordinate) -> float:
        c = Coordinate(*coordinate)
        return np.sqrt(float(self.__x - c.__x)**2 + float(self.__y - c.__y)**2)

    @property
    def x(self) -> float:
        return float(self.__x)

    @property
    def y(self) -> float:
        return float(self.__y)

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
        return arrange(self.__start - padding, self.__stop + padding, step, center)


T = tp.TypeVar("T")


class Grid(tp.Generic[T]):

    def __init__(self, xs: tp.Sequence[float], ys: tp.Sequence[float]) -> None:
        self.__width = len(xs)
        self.__height = len(ys)

        self.__grid = np.full((self.__width, self.__height), fill_value=None)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                self.__grid[i][j] = Coordinate(x, y)

    def __getattr__(self, item) -> np.ndarray:
        # pylint: disable=protected-access
        grid = np.full(self.shape, fill_value=np.nan)

        for x in range(self.__width):
            for y in range(self.__height):
                if self.__grid[x][y] is not None:
                    grid[x][y] = self.__grid[x][y].__getattribute__(item)

        return grid

    def __iter__(self) -> tp.Iterator[T]:
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

    def flatten(self) -> list[T]:
        return list(self.__grid.flatten())

    def transform(self, mesh: Mesh[T]) -> Grid[T]:
        # pylint: disable=protected-access
        copy = Grid([], [])
        copy.__width = self.__width
        copy.__height = self.__height
        copy.__grid = np.full(copy.shape, fill_value=None)

        for x in range(copy.__width):
            for y in range(copy.__height):
                c = Coordinate(*self.__grid[x][y])
                if c in mesh:
                    copy.__grid[x][y] = mesh[c]

        return copy

    def mesh(self) -> Mesh:
        mesh = Mesh()
        for c in self:
            mesh.insert(c)
        return mesh


class Mesh(tp.Generic[T]):

    def __init__(self, value_type: type[T] | None = None) -> None:
        self._value_type = value_type
        self.__mesh: dict[Coordinate, T] = {}

    def __contains__(self, coordinate: tuple[float, float] | Coordinate) -> bool:
        return Coordinate(*coordinate) in self.__mesh.keys()

    def __getitem__(self, coordinate: tuple[float, float] | Coordinate) -> T:
        c = Coordinate(*coordinate)
        if c not in self:
            raise KeyError(c)
        return self.__mesh[c]

    def grid(self) -> Grid:
        return Grid(sorted({i.x for i in self.__mesh.keys()}), sorted({i.y for i in self.__mesh.keys()}))

    def keys(self) -> list[Coordinate]:
        return list(self.__mesh.keys())

    def __iter__(self) -> tp.Iterator[tuple[Coordinate, T]]:
        return iter(self.__mesh.items())

    def __len__(self) -> int:
        return len(self.__mesh)

    def __repr__(self) -> str:
        return f'{self.__mesh}'

    def __str__(self) -> str:
        return self.__repr__()

    def __sub__(self, other: Mesh[T]) -> Mesh[T]:
        mesh = Mesh(self._value_type)
        for k, v in self:
            if v is not None:
                mesh.insert(k, v - other[k])
        return mesh

    def __add__(self, other: Mesh[T]) -> Mesh[T]:
        mesh = Mesh(self._value_type)
        for k, v in self:
            if v is not None:
                mesh.insert(k, v)
        for k, v in other:
            if v is not None:
                mesh.insert(k, other[k])
        return mesh

    def insert(self, coordinate: tuple[float, float] | Coordinate, value: T | None = None) -> T:
        c = Coordinate(*coordinate)
        if c in self.__mesh:
            raise KeyError(c)
        self.__mesh[c] = value
        return self.__mesh[c]

    @abstractmethod
    def emplace(self, key: tuple | Coordinate, **kwargs) -> T:
        assert self._value_type is not None
        return self.insert(key, self._value_type(**kwargs))

    def clear(self) -> None:
        self.__mesh.clear()

    def copy(self) -> Mesh[T]:
        # pylint: disable=protected-access
        mesh = Mesh(self._value_type)
        mesh.__mesh = self.__mesh.copy()
        return mesh

    def detach(self) -> list[tuple[Coordinate, T]]:
        return list(self.__mesh.copy().items())

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for k, v in self.__mesh.items():
                c = ",".join(f'{i:.3f}' for i in k)
                r = ",".join(f'{i:.16f}' for i in v)
                f.write(f'{c},{r}\n')

    def load(self, path: Path) -> None:
        if path.exists():
            lines = path.read_text(encoding='utf-8').strip().split('\n')
            for line in lines:
                token = line.split(',')
                self.insert((float(token[0]), float(token[1])), self._value_type(*[float(i) for i in token[2:]]))
