from __future__ import annotations

import typing as tp
from abc import abstractmethod

import numpy as np

from src.base.geometry import Coordinate, arrange, merge, equal


class Shape:

    @abstractmethod
    def __getitem__(self, step: float) -> tp.Sequence[Coordinate]:
        ...

    @abstractmethod
    def __add__(self, summand: float) -> Shape:
        ...

    def __sub__(self, summand: float) -> Shape:
        return self + -summand

    def __contains__(self, coordinate: tuple | Coordinate) -> bool:
        return False


class Cylinder(Shape):

    def __init__(self, center: tuple[float, float] | Coordinate, radius: float) -> None:
        self.__center = center
        self.__radius = radius

    def __add__(self, summand: float) -> Cylinder:
        return Cylinder(self.__center, self.__radius + summand)

    def __getitem__(self, step: float) -> tp.Sequence[Coordinate]:
        return [
            Coordinate(x, np.sin(np.arccos(x)))
            for x in arrange(self.__center.x - self.__radius, self.__center.x + self.__radius, step)
        ]


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

    def __getitem__(self, step: float) -> tp.List[Coordinate]:
        if equal(self.__a.x, self.__b.x):
            return [Coordinate(self.__a.x, y) for y in arrange(self.__a.y, self.__b.y, step)]
        m = (self.__b.y - self.__a.y) / (self.__b.x - self.__a.x)
        return [Coordinate(x, self.__a.y + m * x) for x in arrange(self.__a.x, self.__b.x, step)]


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
        return self.__a.x <= c.x <= self.__b.x and self.__a.y <= c.y <= self.__b.y

    def __add__(self, summand: float) -> Rectangle:
        return Rectangle((self.__a.x - summand, self.__a.y - summand), (self.__b.x + summand, self.__b.y + summand))

    def __getitem__(self, step: float) -> tp.List[Coordinate]:
        left = [Coordinate(self.__a.x, y) for y in arrange(self.__a.y, self.__b.y, step)]
        top = [Coordinate(x, self.__b.y) for x in arrange(self.__a.x, self.__b.x, step)]
        right = [Coordinate(self.__b.x, y) for y in arrange(self.__b.y, self.__a.y, step)]
        bottom = [Coordinate(x, self.__a.y) for x in arrange(self.__b.x, self.__a.x, step)]
        return merge(left, top, right, bottom)


class Airfoil(Shape):
    __m = .02
    __p = .4
    __t = .12

    def __init__(
        self,
        support: tuple[float, float] | Coordinate,
        size: tuple[float, float] | Coordinate,
        angle: float = 0.,
    ) -> None:
        """
        :param size: size of shape
        """
        self.__a = Coordinate(*support)
        self.__size = Coordinate(*size)
        self.__angle = 0

    def __add__(self, summand: float) -> Airfoil:
        return Airfoil(self.__a - (summand / 2, 0), self.__size + (summand, summand / self.__t / 2), self.__angle)

    def __f(self, x) -> tuple[Coordinate, Coordinate]:
        y_t = 5 * self.__t * (.2969 * np.sqrt(x) - .1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        if x <= self.__p:
            y_c = self.__m / self.__p**2 * (2 * self.__p * x - x**2)
            y_x = 2 * self.__m / self.__p**2 * (self.__p - x)
        else:
            y_c = self.__m / (1 - self.__p)**2 * ((1 - 2 * self.__p) + 2 * self.__p * x - x**2)
            y_x = 2 * self.__m / (1 - self.__p)**2 * (self.__p - x)

        upper = Coordinate(x - y_t * np.sin(y_x), y_c + y_t * np.cos(y_x))
        lower = Coordinate(x + y_t * np.sin(y_x), y_c - y_t * np.cos(y_x))

        return upper, lower

    def __contains__(self, coordinate: tuple[float, float] | Coordinate) -> bool:
        c = ((Coordinate(*coordinate) - self.__a) / self.__size.x).rotate(-self.__angle)
        if 0 <= c.x <= 1:
            upper, lower = self.__f(c.x)
            return lower.y <= c.y <= upper.y
        return False

    def __getitem__(self, step: float) -> tp.Sequence[Coordinate]:
        top = []
        bottom = []
        for x in arrange(0, 1, step / self.__size.x):
            upper, lower = self.__f(x)
            top.append(self.__a + upper * self.__size)
            bottom.append(self.__a + lower * self.__size)
        return merge(bottom[::-1], top)
