from __future__ import annotations

import typing as t
from abc import abstractmethod

import numpy as np

EPS = 1e-6


def clamp(f: float) -> int:
    return int(f / EPS)


def arrange(start: float, stop: float, step: float) -> np.ndarray:
    start, stop, step = clamp(start), clamp(stop), clamp(step)
    r = []
    while start <= stop:
        r.append(start * EPS)
        start += step
    return np.array(r)


class Coordinate:

    def __init__(self, x: float | np.ndarray, y: float | np.ndarray) -> None:
        self.__x = x
        self.__y = y

    def __eq__(self, other) -> bool:
        return clamp(self.x) == clamp(other.x) and clamp(self.y) == clamp(other.y)

    def __getitem__(self, key) -> float:
        return [self.x, self.y][key]

    def __hash__(self) -> int:
        return hash((clamp(self.x), clamp(self.y)))

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


class Axis:

    def __init__(self, label: str, start: float, stop: float) -> None:
        self.__start = start
        self.__stop = stop
        self.__label = label

    @property
    def dim(self) -> t.Tuple[float, float]:
        return self.__start, self.__stop

    @property
    def start(self) -> float:
        return self.__start

    @property
    def stop(self) -> float:
        return self.__stop

    @property
    def label(self) -> str:
        return self.__label

    def arrange(self, step: float, center=False) -> np.ndarray:
        assert step > 0
        c = step / 2 if center else 0
        return arrange(self.__start + c, self.__stop - c, step)


class Mesh:

    def __init__(self, xi: t.Sequence[int | float] = (), yi: t.Sequence[int | float] = ()) -> None:
        self.__width = len(xi)
        self.__height = len(yi)

        self.__mesh = np.zeros((self.__width, self.__height), dtype="object")
        for i, x in enumerate(xi):
            for j, y in enumerate(yi):
                self.__mesh[i][j] = Coordinate(x, y)

    def __getattr__(self, item) -> t.Any:
        return self.map(lambda i: i.__getattribute__(item))

    def __getitem__(self, key) -> t.Any:
        if len(key) == 2:
            return self.__mesh[key].flatten()
        x, y, z, *_ = key
        return self.map(lambda i: i[z])[x, y]

    def __iter__(self) -> t.Iterator:
        return iter(self.__mesh.flatten())

    def __len__(self) -> int:
        return self.__mesh.size

    def __repr__(self) -> str:
        return f'{self.__mesh}'

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def height(self) -> int:
        return self.__height

    @property
    def shape(self) -> t.Tuple[int, int]:
        return self.__width, self.__height

    @property
    def width(self) -> int:
        return self.__width

    def flatten(self) -> t.List[Coordinate]:
        return list(self.__mesh.flatten())

    def numpy(self) -> np.ndarray:
        return self.__mesh.copy()

    def map(self, f) -> Mesh:
        copy = Mesh()
        copy.__mesh = np.array(list(map(lambda i: np.array(list(map(f, i))), self.__mesh.copy())))
        return copy


class Cloud:

    def __init__(self) -> None:
        self.__cloud: t.Dict[Coordinate, t.Any] = dict()

    def __contains__(self, key: t.Tuple | Coordinate) -> bool:
        return Coordinate(*key) in self.__cloud.keys()

    def __getitem__(self, key: t.Tuple | Coordinate) -> t.Any:
        k = Coordinate(*key)
        if k not in self:
            raise KeyError(k)
        return self.__cloud[k]

    def __iter__(self):
        return iter(self.__cloud.items())

    def __len__(self) -> int:
        return len(self.__cloud)

    def __repr__(self) -> str:
        return f'{self.__cloud}'

    def __str__(self) -> str:
        return self.__repr__()

    def add(self, key: t.Tuple | Coordinate, value: t.Any) -> t.Any:
        k = Coordinate(*key)
        if k in self.__cloud:
            raise KeyError(k)
        self.__cloud[k] = value
        return self.__cloud[k]

    def clear(self) -> None:
        self.__cloud.clear()

    def copy(self) -> t.Any:
        cloud = Cloud()
        cloud.__cloud = dict(self.__cloud)
        return cloud

    def detach(self) -> t.List[t.Tuple[Coordinate, t.Any]]:
        return list(self.__cloud.copy().items())

    def numpy(self) -> np.ndarray:
        return np.array([np.concatenate([k.numpy(), v.numpy()]) for k, v in self.__cloud.items()])


class Shape:

    @property
    @abstractmethod
    def x(self) -> t.List[float]:
        ...

    @property
    @abstractmethod
    def y(self) -> t.List[float]:
        ...


class Line(Shape):

    def __init__(self, a: t.Tuple[float, float] | Coordinate, b: t.Tuple[float, float] | Coordinate):
        self.__a = Coordinate(*a)
        self.__b = Coordinate(*b)

    @property
    def x(self) -> t.List[float]:
        return [self.__a.x, self.__b.x]

    @property
    def y(self) -> t.List[float]:
        return [self.__a.y, self.__b.y]


class Rectangle(Shape):

    def __init__(self, a: t.Tuple[float, float] | Coordinate, b: t.Tuple[float, float] | Coordinate):
        self.__a = Coordinate(*a)
        self.__b = Coordinate(*b)

    @property
    def x(self) -> t.List[float]:
        return [self.__a.x, self.__a.x, self.__b.x, self.__b.x, self.__a.x]

    @property
    def y(self) -> t.List[float]:
        return [self.__a.y, self.__b.y, self.__b.y, self.__a.y, self.__a.y]
