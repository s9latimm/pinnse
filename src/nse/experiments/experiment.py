import typing as t

import numpy as np

from src import HIRES
from src.base.data import Mesh, Axis, Shape
from src.foam import get_foam
from src.nse.data import NSECloud


class NSEExperiment:

    def __init__(
            self,
            name: str,
            x: Axis,
            y: Axis,
            mesh: float,
            nu: float,
            rho: float,
            flow: float,
            foam: bool,
            supervised: bool,
            geometry: t.List[Shape] = (),
    ) -> None:
        self.__name = name
        self.__x = x
        self.__y = y
        self.__nu = nu
        self.__rho = rho
        self.__flow = flow
        self.__supervised = supervised
        self.__geometry = geometry

        self._training_mesh = Mesh(self.__x.arrange(mesh / 2), self.__y.arrange(mesh / 2))
        self._finite_elements_mesh = Mesh(self.__x.arrange(mesh, True), self.__y.arrange(mesh, True))
        self._evaluation_mesh = Mesh(self.__x.arrange(mesh / HIRES, True), self.__y.arrange(mesh / HIRES, True))

        self._pde_facts = NSECloud()
        self._rim_facts = NSECloud()

        if foam or supervised:
            self._foam_facts = self.__foam()

    @property
    def pde_facts(self) -> NSECloud:
        return self._pde_facts

    @property
    def rim_facts(self) -> NSECloud:
        return self._rim_facts

    @property
    def training_mesh(self) -> Mesh:
        return self._training_mesh

    @property
    def finite_elements_mesh(self) -> Mesh:
        return self._finite_elements_mesh

    @property
    def evaluation_mesh(self) -> Mesh:
        return self._evaluation_mesh

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
    def foam_facts(self) -> NSECloud:
        return self._foam_facts

    @property
    def geometry(self) -> t.List[Shape]:
        return self.__geometry

    @property
    def dim(self) -> t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]:
        return self.__x.dim, self.__y.dim

    def __foam(self) -> NSECloud:
        u, v, p = get_foam()
        u = np.flip(u, 0).transpose().flatten()
        v = np.flip(v, 0).transpose().flatten()
        p = np.flip(p, 0).transpose().flatten()

        foam_cloud = NSECloud()

        for i, c in enumerate(self._finite_elements_mesh):
            foam_cloud.add(c, u=u[i], v=v[i], p=p[i])

        return foam_cloud
