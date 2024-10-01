import typing as t

import matplotlib.pyplot as plt
from matplotlib import patches

from src import HIRES
from src.base.data import arrange, Grid
from src.nse.data import NSECloud


class Axis:

    def __init__(self, start: float, stop: float, label: str) -> None:
        self.__start = start
        self.__stop = stop
        self.__label = label

    @property
    def start(self) -> float:
        return self.__start

    @property
    def stop(self) -> float:
        return self.__stop

    @property
    def label(self) -> str:
        return self.__label


class NSEExperiment:

    def __init__(
            self,
            name: str,
            x: Axis,
            y: Axis,
            grid: float,
            nu: float,
            rho: float,
            flow: float,
            overlay: t.List[patches.Patch] = (),
    ) -> None:
        self.__name = name
        self.__x = x
        self.__y = y
        self.__nu = nu
        self.__rho = rho
        self.__flow = flow

        self.__overlay = overlay

        step = grid / 2
        self.layout_grid = Grid(
            arrange(self.x.start, self.x.stop, step),
            arrange(self.y.start, self.y.stop, step),
        )

        step = grid
        off = step / 2
        self.finite_grid = Grid(
            arrange(self.x.start + off, self.x.stop - off, step),
            arrange(self.y.start + off, self.y.stop - off, step),
        )

        step = grid / HIRES
        off = step / 2
        self.hires_grid = Grid(
            arrange(self.x.start + off, self.x.stop - off, step),
            arrange(self.y.start + off, self.y.stop - off, step),
        )

        self.__pde = NSECloud()
        self.__rim = NSECloud()

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

    def apply(self, ax: plt.Axes) -> None:
        for patch in self.__overlay:
            ax.add_patch(patch)
