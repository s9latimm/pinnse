import numpy as np

from src.base.geometry import Axis, Mesh
from src.base.shapes import Shape
from src.foam import get_foam
from src.nse.data import NSECloud


class NSEExperiment:

    def __init__(
            self,
            name: str,
            x: Axis,
            y: Axis,
            nu: float,
            rho: float,
            inlet: float,
            foam: bool,
            supervised: bool,
            geometry: list[Shape] = (),
    ) -> None:
        self.__name = name
        self.__x = x
        self.__y = y
        self.__nu = nu
        self.__rho = rho
        self.__inlet = inlet
        self.__supervised = supervised
        self.__geometry = geometry

        self._learning = NSECloud()
        self._knowledge = NSECloud()

        if foam or supervised:
            self._foam_facts = self.__foam()

    @property
    def learning(self) -> NSECloud:
        return self._learning

    @property
    def knowledge(self) -> NSECloud:
        return self._knowledge

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
    def foam_facts(self) -> NSECloud:
        return self._foam_facts

    @property
    def geometry(self) -> list[Shape]:
        return self.__geometry

    @property
    def dim(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.__x.dim, self.__y.dim

    def __foam(self) -> NSECloud:
        u, v, p = get_foam()
        u = np.flip(u, 0).transpose().flatten()
        v = np.flip(v, 0).transpose().flatten()
        p = np.flip(p, 0).transpose().flatten()

        foam_cloud = NSECloud()

        mesh = Mesh(self.__x.arrange(.1, True), self.__y.arrange(.1, True))

        for i, c in enumerate(mesh):
            foam_cloud.add(c, u=u[i], v=v[i], p=p[i])

        return foam_cloud
