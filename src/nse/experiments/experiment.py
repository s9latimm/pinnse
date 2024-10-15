import typing as tp

from src.base.function import Function, Null
from src.base.mesh import Axis
from src.base.shape import Figure
from src.nse.data import NSEMesh

Foam: tp.TypeAlias = 'Foam'
# type Foam = 'Foam'


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
            inlet: Function = Null(),
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
        self.__supervised = supervised

        self._in = inlet

        self._learning = NSEMesh()
        self._knowledge = NSEMesh()
        self._evaluation = NSEMesh()
        self._inlet = NSEMesh()
        self._outlet = NSEMesh()

        if foam or supervised:
            self.__foam = foam

    @property
    def learning(self) -> NSEMesh:
        return self._learning

    @property
    def knowledge(self) -> NSEMesh:
        return self._knowledge

    @property
    def evaluation(self) -> NSEMesh:
        return self._evaluation

    @property
    def inlet_f(self) -> Function:
        return self._in

    @property
    def inlet(self) -> NSEMesh:
        return self._inlet

    @property
    def outlet(self) -> NSEMesh:
        return self._outlet

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
    def shape(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.__x.shape, self.__y.shape

    @property
    def nu(self) -> float:
        return self.__nu

    @property
    def rho(self) -> float:
        return self.__rho

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
