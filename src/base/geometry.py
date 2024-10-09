from __future__ import annotations

import typing as tp

import numpy as np

EPS: float = 1e-6


def clamp(f: float) -> int:
    return int(f / EPS)


def merge(*lists: tp.Sequence[tp.Any]) -> list[tp.Any]:
    merged = []
    for xs in lists:
        for x in xs:
            if x not in merged:
                merged.append(x)
    return merged


def equal(a: float, b: float) -> bool:
    return clamp(a) == clamp(b)


def arrange(start: float, stop: float, step: float) -> list[float]:
    start, stop, step = clamp(start), clamp(stop), clamp(step)
    r = []
    if step <= 0.:
        return r
    if start < stop:
        while start <= stop:
            r.append(start * EPS)
            start += step
    elif start > stop:
        while start >= stop:
            r.append(start * EPS)
            start -= step
    else:
        return [start * EPS]
    return r


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

    def __iter__(self) -> tp.Iterator[float]:
        return iter((self.x, self.y))

    def __repr__(self) -> str:
        return f'Coordinate(x={str(self.x)}, y={str(self.y)})'

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, coordinate: tuple[float, float] | Coordinate) -> Coordinate:
        c = Coordinate(*coordinate)
        return Coordinate(self.x + c.x, self.y + c.y)

    def __mul__(self, coordinate: tuple[float, float] | Coordinate) -> Coordinate:
        c = Coordinate(*coordinate)
        return Coordinate(self.x * c.x, self.y * c.y)

    def __sub__(self, coordinate: tuple[float, float] | Coordinate) -> Coordinate:
        c = Coordinate(*coordinate)
        return Coordinate(self.x - c.x, self.y - c.y)

    def __truediv__(self, factor: float) -> Coordinate:
        if equal(factor, 0):
            return Coordinate(np.infty, np.infty)
        return Coordinate(self.x / factor, self.y / factor)

    def numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])

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
    def label(self) -> str:
        return self.__label

    def arrange(self, step: float, center=False, padding=0) -> list[float]:
        assert step > 0
        c = step / 2 if center else 0
        return arrange(self.__start + c - padding, self.__stop - c + padding, step)


class Mesh:

    def __init__(self, xs: tp.Sequence[float], ys: tp.Sequence[float]) -> None:
        self.__width = len(xs)
        self.__height = len(ys)

        self.__mesh = np.zeros((self.__width, self.__height), dtype="object")
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                self.__mesh[i][j] = Coordinate(x, y)

    def __getattr__(self, item) -> np.ndarray:
        return self.map(lambda i: i.__getattribute__(item)).__mesh

    def __iter__(self) -> tp.Iterator:
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
    def shape(self) -> tuple[int, int]:
        return self.__width, self.__height

    @property
    def width(self) -> int:
        return self.__width

    def flatten(self) -> list[Coordinate]:
        return list(self.__mesh.flatten())

    def transform(self, cloud: Cloud):
        return self.map(lambda i: cloud[i])

    def map(self, f) -> Mesh:
        copy = Mesh([], [])
        copy.__mesh = np.array(list(map(lambda i: np.array(list(map(f, i))), self.__mesh.copy())))
        copy.__width = self.__width
        copy.__height = self.__height
        return copy


class Cloud:

    def __init__(self) -> None:
        self.__cloud: dict[Coordinate, tp.Any] = dict()

    def __contains__(self, coordinate: tuple[float, float] | Coordinate) -> bool:
        return Coordinate(*coordinate) in self.__cloud.keys()

    def __getitem__(self, coordinate: tuple[float, float] | Coordinate) -> tp.Any:
        c = Coordinate(*coordinate)
        if c not in self:
            raise KeyError(c)
        return self.__cloud[c]

    def mesh(self, refine=0) -> Mesh:
        xs = sorted({i.x for i in self.__cloud.keys()})
        ys = sorted({i.y for i in self.__cloud.keys()})

        # if refine > 0:
        #     xs = (xs[:, None] + np.linspace(0., 1., refine)).ravel()
        #     print(xs)
        #     ys = (ys[:, None] + np.linspace(0., 1., refine)).ravel()

        return Mesh(xs, ys)

    def keys(self) -> list[Coordinate]:
        return list(self.__cloud.keys())

    def __iter__(self) -> tp.Iterator[tuple[Coordinate, tp.Any]]:
        return iter(self.__cloud.items())

    def __len__(self) -> int:
        return len(self.__cloud)

    def __repr__(self) -> str:
        return f'{self.__cloud}'

    def __str__(self) -> str:
        return self.__repr__()

    def add(self, coordinate: tuple[float, float] | Coordinate, value: tp.Any) -> tp.Any:
        c = Coordinate(*coordinate)
        if c in self.__cloud:
            raise KeyError(c)
        self.__cloud[c] = value
        return self.__cloud[c]

    def clear(self) -> None:
        self.__cloud.clear()

    def copy(self) -> tp.Any:
        cloud = Cloud()
        cloud.__cloud = self.__cloud.copy()
        return cloud

    def detach(self) -> list[tuple[Coordinate, tp.Any]]:
        return list(self.__cloud.copy().items())

    def numpy(self) -> np.ndarray:
        return np.array([np.concatenate([k.numpy(), v.numpy()]) for k, v in self.__cloud.items()])
