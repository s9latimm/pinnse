from __future__ import annotations

import typing as tp
from abc import abstractmethod

import numpy as np

EPS = 1e-6


def clamp(f: float) -> int:
    return int(f / EPS)


def equal(a: float, b: float) -> bool:
    return clamp(a) == clamp(b)


def arrange(start: float, stop: float, step: float) -> np.ndarray:
    start, stop, step = clamp(start), clamp(stop), clamp(step)
    r = []
    while start <= stop:
        r.append(start * EPS)
        start += step
    return np.array(r)


def interpolate(
    c: Coordinate,
    fun: tp.Sequence[Coordinate],
    transpose=False,
    extrude: float = 2,
) -> tp.Optional[Coordinate]:
    if not transpose:
        left = sorted([i for i in fun if i.x <= c.x], key=lambda i: i.x, reverse=True)
        right = sorted([i for i in fun if i.x > c.x], key=lambda i: i.x)
        if len(left) > 0 and len(right) > 0:
            left = left[0]
            right = right[0]

            return Coordinate(
                c.x,
                left.y + (c.x - left.x) * (right.y - left.y) / (right.x - left.x),
            )
    else:
        down = sorted([i for i in fun if i.y <= c.y], key=lambda i: i.y, reverse=True)
        up = sorted([i for i in fun if i.y > c.y], key=lambda i: i.y)
        if len(down) > 0 and len(up) > 0:
            down = down[0]
            up = up[0]

            return interpolate(Coordinate(
                down.x + (c.y - down.y) * (up.x - down.x) / (up.y - down.y),
                c.y,
            ), fun)
    return None


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
    def dim(self) -> tp.Tuple[float, float]:
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

    def __init__(self, xi: tp.Sequence[int | float] = (), yi: tp.Sequence[int | float] = ()) -> None:
        self.__width = len(xi)
        self.__height = len(yi)

        self.__mesh = np.zeros((self.__width, self.__height), dtype="object")
        for i, x in enumerate(xi):
            for j, y in enumerate(yi):
                self.__mesh[i][j] = Coordinate(x, y)

    def __getattr__(self, item) -> tp.Any:
        return self.map(lambda i: i.__getattribute__(item))

    def __getitem__(self, key) -> tp.Any:
        if len(key) == 2:
            return self.__mesh[key].flatten()
        x, y, z, *_ = key
        return self.map(lambda i: i[z])[x, y]

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
    def shape(self) -> tp.Tuple[int, int]:
        return self.__width, self.__height

    @property
    def width(self) -> int:
        return self.__width

    def flatten(self) -> tp.List[Coordinate]:
        return list(self.__mesh.flatten())

    def numpy(self) -> np.ndarray:
        return self.__mesh.copy()

    def map(self, f) -> Mesh:
        copy = Mesh()
        copy.__mesh = np.array(list(map(lambda i: np.array(list(map(f, i))), self.__mesh.copy())))
        return copy


class Cloud:

    def __init__(self) -> None:
        self.__cloud: tp.Dict[Coordinate, tp.Any] = dict()

    def __contains__(self, key: tp.Tuple | Coordinate) -> bool:
        return Coordinate(*key) in self.__cloud.keys()

    def __getitem__(self, key: tp.Tuple | Coordinate) -> tp.Any:
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

    def add(self, key: tp.Tuple | Coordinate, value: tp.Any) -> tp.Any:
        k = Coordinate(*key)
        if k in self.__cloud:
            raise KeyError(k)
        self.__cloud[k] = value
        return self.__cloud[k]

    def clear(self) -> None:
        self.__cloud.clear()

    def copy(self) -> tp.Any:
        cloud = Cloud()
        cloud.__cloud = dict(self.__cloud)
        return cloud

    def detach(self) -> tp.List[tp.Tuple[Coordinate, tp.Any]]:
        return list(self.__cloud.copy().items())

    def numpy(self) -> np.ndarray:
        return np.array([np.concatenate([k.numpy(), v.numpy()]) for k, v in self.__cloud.items()])


class Shape:

    @property
    @abstractmethod
    def x(self) -> tp.List[float]:
        ...

    @property
    @abstractmethod
    def y(self) -> tp.List[float]:
        ...


class Line(Shape):

    def __init__(self, a: tp.Tuple[float, float] | Coordinate, b: tp.Tuple[float, float] | Coordinate):
        self.__a = Coordinate(*a)
        self.__b = Coordinate(*b)

    @property
    def x(self) -> tp.List[float]:
        return [self.__a.x, self.__b.x]

    @property
    def y(self) -> tp.List[float]:
        return [self.__a.y, self.__b.y]


class Rectangle(Shape):

    def __init__(self, a: tp.Tuple[float, float] | Coordinate, b: tp.Tuple[float, float] | Coordinate):
        self.__a = Coordinate(*a)
        self.__b = Coordinate(*b)

    @property
    def x(self) -> tp.List[float]:
        return [self.__a.x, self.__a.x, self.__b.x, self.__b.x, self.__a.x]

    @property
    def y(self) -> tp.List[float]:
        return [self.__a.y, self.__b.y, self.__b.y, self.__a.y, self.__a.y]


class Airfoil(Shape):

    def __init__(self, a: tp.Tuple[float, float] | Coordinate, b: tp.Tuple[float, float] | Coordinate):
        self.__a = Coordinate(*a)
        self.__b = Coordinate(*b)
        self.__angle = -10

        self.__m = .02
        self.__p = .4
        self.__t = .12

        f_u = []
        f_l = []

        for x in arrange(0, 1, .005):
            upper, lower = self._f(x)
            f_u.append(upper)
            f_l.append(lower)

        p_u = set()
        p_l = set()
        for x in arrange(0, 20, .05):
            for y in arrange(0, 2, .05):
                c = Coordinate(x, y)
                p_u.add(interpolate(c, f_u))
                p_u.add(interpolate(c, f_u, True))
                p_l.add(interpolate(c, f_l))
                p_l.add(interpolate(c, f_l, True))

        p_u = [i for i in p_u if i is not None]
        p_l = [i for i in p_l if i is not None]

        self.__f = sorted(p_u, key=lambda i: i.x, reverse=True) + sorted(p_l, key=lambda i: i.x)

    def _f(self, x) -> tp.Tuple[Coordinate, Coordinate]:
        if x <= self.__p:
            y_c = self.__m / self.__p**2 * (2 * self.__p * x - x**2)
            y_x = 2 * self.__m / self.__p**2 * (self.__p - x)
            y_t = 5 * self.__t * (.2969 * np.sqrt(x) - .1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        else:
            y_c = self.__m / (1 - self.__p)**2 * ((1 - 2 * self.__p) + 2 * self.__p * x - x**2)
            y_x = 2 * self.__m / (1 - self.__p)**2 * (self.__p - x)
            y_t = 5 * self.__t * (.2969 * np.sqrt(x) - .1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

        return Coordinate(
            self.__a.x + self.__b.x * (x - y_t * np.sin(y_x)),
            self.__a.y + self.__b.x * (y_c + y_t * np.cos(y_x)),
        ).rotate(self.__angle), Coordinate(
            self.__a.x + self.__b.x * (x + y_t * np.sin(y_x)),
            self.__a.y + self.__b.x * (y_c - y_t * np.cos(y_x)),
        ).rotate(self.__angle)

    def __contains__(self, key: tp.Tuple | Coordinate) -> bool:
        k = Coordinate(*key)
        p = sorted([i for i in self.__f if equal(k.x, i.x)], key=lambda i: i.y)
        if len(p) > 0:
            return p[0].y <= k.y <= p[-1].y
        return False

    @property
    def x(self) -> tp.List[float]:
        return [i.x for i in self.__f]

    @property
    def y(self) -> tp.List[float]:
        return [i.y for i in self.__f]
