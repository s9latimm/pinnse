from abc import abstractmethod

import numpy as np


class Function:

    @abstractmethod
    def __call__(self, x: float) -> float:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        return self.__str__()


class Constant(Function):

    def __init__(self, constant: float) -> None:
        self.__constant = constant

    def __call__(self, x: float) -> float:
        return self.__constant

    def __str__(self) -> str:
        return f'{self.__constant:.0f}'


class Null(Constant):

    def __init__(self) -> None:
        super().__init__(0)


class Sinus(Function):

    def __init__(self, start: float, end: float, scale: float = 1.) -> None:
        self.__scale = scale
        self.__a = 2 * np.pi * (end - start)

    def __call__(self, x: float) -> float:
        return self.__scale * np.sin(self.__a * x)

    def __str__(self) -> str:
        return f'{self.__scale:.1f}*sin({self.__a:.1f}x)'


class Parabola(Function):

    def __init__(self, start: float, end: float, scale: float = 1.) -> None:
        self.__scale = scale
        self.__width = end - start
        self.__a = 2. / self.__width
        self.__b = (end - self.__width / 2.) * self.__a

    def __call__(self, x: float) -> float:
        return self.__scale * self.__width * (1. - (self.__b - self.__a * x)**2)

    def __str__(self) -> str:
        return f'{self.__scale:.1f}*{self.__width:.1f}*(1-({self.__b:.1f}-{self.__a:.1f}x)^2)'
