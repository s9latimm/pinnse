import typing as tp

from src.base.mesh import Axis
from src.base.shape import Figure
from src.nse.data import NSECloud

Foam: tp.TypeAlias = 'Foam'
# type Foam = 'Foam'


def inlet(start: float, end: float, scale: float = 1.) -> tp.Callable[[float], float]:
    width = end - start
    mid = end - width / 2.

    b = 2. / width
    a = mid * b

    return lambda x: scale * (1. - (a - b * x)**2)


class NSEExperiment:

    def __init__(
        self,
        name: str,
        x: Axis,
        y: Axis,
        boundary: Figure = None,
        obstruction: Figure = None,
        nu: float = 1,
        rho: float = 1,
        flow: float = 1,
        foam: Foam = None,
        supervised: bool = False,
    ) -> None:
        self.__name = name
        self.__x = x
        self.__y = y
        self.__boundary = boundary
        self.__obstruction = obstruction
        self.__nu = nu
        self.__rho = rho
        self.__flow = flow
        self.__supervised = supervised

        self._learning = NSECloud()
        self._knowledge = NSECloud()
        self._evaluation = NSECloud()

        if foam or supervised:
            self.__foam = foam

    @property
    def learning(self) -> NSECloud:
        return self._learning

    @property
    def knowledge(self) -> NSECloud:
        return self._knowledge

    @property
    def evaluation(self) -> NSECloud:
        return self._evaluation

    @property
    def name(self) -> str:
        return self.__name

    @property
    def x(self) -> Axis:
        return self.__x

    @property
    def y(self) -> Axis:
        return self.__y

    @property
    def nu(self) -> float:
        return self.__nu

    @property
    def rho(self) -> float:
        return self.__rho

    @property
    def flow(self) -> float:
        return self.__flow

    @property
    def supervised(self) -> bool:
        return self.__supervised

    @property
    def foam(self) -> Foam:
        return self.__foam

    @property
    def boundary(self) -> Figure:
        return self.__boundary

    @property
    def obstruction(self) -> Figure:
        return self.__obstruction

    @property
    def dim(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.__x.dim, self.__y.dim
