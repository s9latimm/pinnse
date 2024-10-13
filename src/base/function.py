from abc import abstractmethod


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
        return f'{self.__constant:.1f}'


class Null(Constant):

    def __init__(self) -> None:
        super().__init__(0)


class Parabola(Function):

    def __init__(self, start: float, end: float, scale: float = 1.) -> None:
        self.__scale = scale
        self.__width = end - start
        mid = end - self.__width / 2.

        self.__b = 2. / self.__width
        self.__a = mid * self.__b

    def __call__(self, x: float) -> float:
        return self.__scale * self.__width * (1. - (self.__a - self.__b * x)**2)

    def __str__(self) -> str:
        return f'{self.__scale:.1f} * {self.__width:.1f} * (1.0 - ({self.__a:.1f} - {self.__b:.1f} * x)**2)'
