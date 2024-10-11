from __future__ import annotations

import typing as tp
from abc import abstractmethod

import numpy as np

from src.base.mesh import Coordinate, arrange, merge, equal, leq


class Shape:

    @abstractmethod
    def __getitem__(self, step: slice) -> Polygon:
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


class Polygon(Shape):

    def __init__(self, *vertices: Coordinate):
        self.__vertices = vertices

    def __getitem__(self, s: slice) -> Polygon:
        coordinates = self.__vertices
        if s.start is not None:
            coordinates = [i for i in coordinates if i.x >= s.start]
        if s.stop is not None:
            coordinates = [i for i in coordinates if i.x <= s.stop]
        return Polygon(*coordinates)

    def __add__(self, summand: float) -> Shape:
        return Polygon(*[i + (summand, summand) for i in self.__vertices])

    def interpolate(self, translation: float = 0.) -> Polygon:
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
        return Polygon(*vertices)

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
        return [i.x for i in self.__vertices]

    @property
    def y(self) -> list[float]:
        return [i.y for i in self.__vertices]


class Cylinder(Shape):

    def __init__(self, center: tuple[float, float] | Coordinate, radius: float) -> None:
        self.__center = center
        self.__radius = radius

    def __add__(self, summand: float) -> Cylinder:
        return Cylinder(self.__center, self.__radius + summand)

    def __getitem__(self, s: slice) -> Polygon:
        coordinates = [
            Coordinate(x, np.sin(np.arccos(x)))
            for x in arrange(self.__center.x - self.__radius, self.__center.x + self.__radius, s.step)
        ]
        if s.start is not None:
            coordinates = [i for i in coordinates if i.x >= s.start]
        if s.stop is not None:
            coordinates = [i for i in coordinates if i.x <= s.stop]
        return Polygon(*coordinates)


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

    def __getitem__(self, s: slice) -> Polygon:
        if equal(self.__a.x, self.__b.x):
            return Polygon(*[Coordinate(self.__a.x, y) for y in arrange(self.__a.y, self.__b.y, s.step)])
        m = (self.__b.y - self.__a.y) / (self.__b.x - self.__a.x)
        coordinates = [Coordinate(x, self.__a.y + m * x) for x in arrange(self.__a.x, self.__b.x, s.step)]
        if s.start is not None:
            coordinates = [i for i in coordinates if i.x >= s.start]
        if s.stop is not None:
            coordinates = [i for i in coordinates if i.x <= s.stop]
        return Polygon(*coordinates)


class Rectangle(Shape):

    def __init__(self, a: tuple[float, float] | Coordinate, b: tuple[float, float] | Coordinate) -> None:
        """
        :param a: lower left corner
        :param b: upper right corner
        """
        self.__a = Coordinate(*a)
        self.__b = Coordinate(*b)

    def __contains__(self, coordinate: tuple | Coordinate) -> bool:
        c = Coordinate(*coordinate)
        return leq(self.__a.x, c.x) and leq(c.x, self.__b.x) and leq(self.__a.y, c.y) and leq(c.y, self.__b.y)

    def __add__(self, summand: float) -> Rectangle:
        return Rectangle((self.__a.x - summand, self.__a.y - summand), (self.__b.x + summand, self.__b.y + summand))

    def __getitem__(self, s: slice) -> Polygon:
        left = [Coordinate(self.__a.x, y) for y in arrange(self.__a.y, self.__b.y, s.step)]
        top = [Coordinate(x, self.__b.y) for x in arrange(self.__a.x, self.__b.x, s.step)]
        right = [Coordinate(self.__b.x, y) for y in arrange(self.__b.y, self.__a.y, s.step)]
        bottom = [Coordinate(x, self.__a.y) for x in arrange(self.__b.x, self.__a.x, s.step)]
        coordinates = merge(left, top, right, bottom)
        if s.start is not None:
            coordinates = [i for i in coordinates if i.x >= s.start]
        if s.stop is not None:
            coordinates = [i for i in coordinates if i.x <= s.stop]
        return Polygon(*coordinates)


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
        c = ((Coordinate(*coordinate) - self.__a) / self.__length)
        if 0 <= c.x <= 1:
            upper, lower = self.__f(c.x)
            return lower.y <= c.y <= upper.y
        return False

    def __getitem__(self, s: slice) -> Polygon:
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
        return Polygon(*coordinates)
