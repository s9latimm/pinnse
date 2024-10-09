from src.base.mesh import Axis
from src.base.shape import Figure
from src.nse.data import NSECloud


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
        inlet: float = 1,
        foam=None,
        supervised: bool = False,
    ) -> None:
        self.__name = name
        self.__x = x
        self.__y = y
        self.__boundary = boundary
        self.__obstruction = obstruction
        self.__nu = nu
        self.__rho = rho
        self.__inlet = inlet
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
    def inlet(self) -> float:
        return self.__inlet

    @property
    def supervised(self) -> bool:
        return self.__supervised

    @property
    def foam(self):
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
