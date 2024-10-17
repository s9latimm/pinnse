from __future__ import annotations

import typing as tp
from abc import abstractmethod

import numpy as np

from src.base.model.mesh import Coordinate, arrange, merge, RealNumber


class Shape:

    @abstractmethod
    def __getitem__(self, step: slice) -> _Polygon:
        ...

    @abstractmethod
    def __add__(self, summand: float) -> Shape:
        ...

    def __sub__(self, summand: float) -> Shape:
        return self + -summand

    def __contains__(self, coordinate: tuple | Coordinate) -> bool:
        return False


class Figure:

    def __init__(self, *shapes: Shape):
        self.__shapes = shapes

    def __iter__(self) -> tp.Iterator[Shape]:
        return iter(self.__shapes)

    def __contains__(self, coordinate: tuple | Coordinate) -> bool:
        c = Coordinate(*coordinate)
        for s in self.__shapes:
            if c in s:
                return True
        return False


class _Polygon(Shape):

    def __init__(self, *vertices: Coordinate, cyclic: bool):
        self.__cyclic = cyclic
        self.__vertices = vertices

    def __getitem__(self, s: slice) -> _Polygon:
        coordinates = self.__vertices
        if s.start is not None:
            coordinates = [i for i in coordinates if i.x >= s.start]
        if s.stop is not None:
            coordinates = [i for i in coordinates if i.x <= s.stop]
        return _Polygon(*coordinates, cyclic=self.__cyclic)

    def __add__(self, summand: float) -> Shape:
        return _Polygon(*[i + (summand, summand) for i in self.__vertices], cyclic=self.__cyclic)

    def interpolate(self, translation: float = 0.) -> _Polygon:
        vertices = []
        for i in range(1, len(self.__vertices)):
            left = self.__vertices[i - 1]
            right = self.__vertices[i]
            x = right.x - left.x
            y = right.y - left.y
            a = translation / np.sqrt(x**2 + y**2)
            c = left + Coordinate(x / 2 - y * a, y / 2 + x * a)
            if translation < 0:
                if c not in self:
                    continue
            vertices.append(c)
        return _Polygon(*vertices, cyclic=self.__cyclic)

    def __contains__(self, coordinate: tuple | Coordinate) -> bool:
        c = Coordinate(*coordinate)
        upper, lower = False, False
        for i in range(1, len(self.__vertices)):
            left = self.__vertices[i - 1]
            right = self.__vertices[i]
            if left.x <= c.x <= right.x:
                y = left.y + (right.y - left.y) / 2
                if y >= c.y:
                    upper = True
                else:
                    lower = True
            if right.x <= c.x <= left.x:
                y = right.y + (left.y - right.y) / 2
                if y <= c.y:
                    lower = True
                else:
                    upper = True
        return upper and lower

    def __iter__(self) -> tp.Iterator[Coordinate]:
        return iter(self.__vertices)

    @property
    def x(self) -> list[float]:
        x = [i.x for i in self.__vertices]
        if self.__cyclic:
            return x[-1:] + x
        return x

    @property
    def y(self) -> list[float]:
        y = [i.y for i in self.__vertices]
        if self.__cyclic:
            return y[-1:] + y
        return y


class Circle(Shape):

    def __init__(self, center: tuple[float, float] | Coordinate, radius: float) -> None:
        self.__center = Coordinate(*center)
        self.__radius = radius

    def __add__(self, summand: float) -> Circle:
        return Circle(self.__center, self.__radius + summand)

    def __contains__(self, coordinate: tuple | Coordinate) -> bool:
        c = Coordinate(*coordinate) - self.__center
        return RealNumber(np.sqrt(c.x**2 + c.y**2)) <= RealNumber(self.__radius)

    def __getitem__(self, s: slice) -> _Polygon:
        n = -((2 * np.pi * self.__radius) // -s.step)
        coordinates = [
            self.__center + Coordinate(self.__radius * np.cos(i), self.__radius * np.sin(i))
            for i in arrange(0, 2 * np.pi, 2 * np.pi / n)
        ]
        return _Polygon(*coordinates, cyclic=True)


class Line(Shape):

    def __init__(self, a: tuple[float, float] | Coordinate, b: tuple[float, float] | Coordinate) -> None:
        """
        :param a: left side coordinate
        :param b: right side coordinate
        """
        self.__a = Coordinate(*a)
        self.__b = Coordinate(*b)

    def __add__(self, summand: float) -> Line:
        return Line((self.__a.x - summand, self.__a.y - summand), (self.__b.x + summand, self.__b.y + summand))

    def __getitem__(self, s: slice) -> _Polygon:
        if RealNumber(self.__a.x) == RealNumber(self.__b.x):
            return _Polygon(*[Coordinate(self.__a.x, y) for y in arrange(self.__a.y, self.__b.y, s.step)])
        m = (self.__b.y - self.__a.y) / (self.__b.x - self.__a.x)
        coordinates = [Coordinate(x, self.__a.y + m * x) for x in arrange(self.__a.x, self.__b.x, s.step)]
        if s.start is not None:
            coordinates = [i for i in coordinates if i.x >= s.start]
        if s.stop is not None:
            coordinates = [i for i in coordinates if i.x <= s.stop]
        return _Polygon(*coordinates, cyclic=False)

    def __contains__(self, coordinate: tuple | Coordinate) -> bool:
        c = Coordinate(*coordinate)
        return RealNumber(self.__a.distance(c) + self.__b.distance(c)) == RealNumber(self.__a.distance(self.__b))


class Rectangle(Shape):

    def __init__(self, a: tuple[float, float] | Coordinate, b: tuple[float, float] | Coordinate) -> None:
        """
        :param a: lower left corner
        :param b: upper right corner
        """
        self.__a = Coordinate(*a)
        self.__b = Coordinate(*b)

    @property
    def shape(self):
        return self.__a, self.__b

    def __contains__(self, coordinate: tuple | Coordinate) -> bool:
        c = Coordinate(*coordinate)
        return RealNumber(self.__a.x) <= c.x <= RealNumber(self.__b.x) and RealNumber(self.__a.y) <= c.y <= RealNumber(
            self.__b.y)

    def __add__(self, summand: float) -> Rectangle:
        return Rectangle((self.__a.x - summand, self.__a.y - summand), (self.__b.x + summand, self.__b.y + summand))

    def __getitem__(self, s: slice) -> _Polygon:
        left = [Coordinate(self.__a.x, y) for y in arrange(self.__a.y, self.__b.y, s.step)]
        top = [Coordinate(x, self.__b.y) for x in arrange(self.__a.x, self.__b.x, s.step)]
        right = [Coordinate(self.__b.x, y) for y in arrange(self.__b.y, self.__a.y, s.step)]
        bottom = [Coordinate(x, self.__a.y) for x in arrange(self.__b.x, self.__a.x, s.step)]
        coordinates = merge(left, top, right, bottom)
        if s.start is not None:
            coordinates = [i for i in coordinates if i.x >= s.start]
        if s.stop is not None:
            coordinates = [i for i in coordinates if i.x <= s.stop]
        return _Polygon(*coordinates, cyclic=True)


class Airfoil(Shape):
    __m = .02
    __p = .4
    __t = .12

    def __init__(
        self,
        support: tuple[float, float] | Coordinate,
        length: float,
        angle: float = 0.,
    ) -> None:
        self.__a = Coordinate(*support)
        self.__length = length
        self.__angle = angle

    def __add__(self, summand: float) -> Airfoil:
        return Airfoil(self.__a - (summand / 2, 0), self.__length + summand, self.__angle)

    def __f(self, x) -> tuple[Coordinate, Coordinate]:
        y_t = 5 * self.__t * (.2969 * np.sqrt(x) - .1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        if x <= self.__p:
            y_c = self.__m / self.__p**2 * (2 * self.__p * x - x**2)
            y_x = 2 * self.__m / self.__p**2 * (self.__p - x)
        else:
            y_c = self.__m / (1 - self.__p)**2 * ((1 - 2 * self.__p) + 2 * self.__p * x - x**2)
            y_x = 2 * self.__m / (1 - self.__p)**2 * (self.__p - x)

        upper = Coordinate(x - y_t * np.sin(y_x), y_c + y_t * np.cos(y_x)).rotate(self.__angle)
        lower = Coordinate(x + y_t * np.sin(y_x), y_c - y_t * np.cos(y_x)).rotate(self.__angle)

        return upper, lower

    def __contains__(self, coordinate: tuple[float, float] | Coordinate) -> bool:
        c = (Coordinate(*coordinate) - self.__a) / self.__length
        if 0 <= RealNumber(c.x) <= 1:
            upper, lower = self.__f(c.x)
            return RealNumber(lower.y) <= RealNumber(c.y + 5e-3) and RealNumber(c.y - 5e-3) <= RealNumber(upper.y)
        return False

    def __getitem__(self, s: slice) -> _Polygon:
        top = []
        bottom = []
        for x in arrange(0, 1, s.step / self.__length):
            upper, lower = self.__f(x)
            top.append(self.__a + upper * self.__length)
            bottom.append(self.__a + lower * self.__length)
        coordinates = merge(bottom[::-1], top)
        if s.start is not None:
            coordinates = [i for i in coordinates if i.x >= s.start]
        if s.stop is not None:
            coordinates = [i for i in coordinates if i.x <= s.stop]
        return _Polygon(*coordinates, cyclic=True)
