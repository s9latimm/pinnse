from matplotlib import patches

from src.base.data import arrange
from src.nse.experiments.experiment import Axis, NSEExperiment


class Step(NSEExperiment):

    def __init__(
        self,
        nu: float,
        rho: float,
        flow: float,
        foam: bool = False,
        supervised: bool = False,
    ):
        self.__nu = nu
        self.__rho = rho

        geometry = [
            patches.Rectangle((0, 0), 1, 1, linewidth=1, linestyle=':', edgecolor='k', facecolor='none', zorder=99),
            patches.Rectangle((0, 0), 10, 2, linewidth=1, linestyle=':', edgecolor='k', facecolor='none', zorder=99),
        ]

        super().__init__(
            'Step',
            Axis(0, 10, 'x'),
            Axis(0, 2, 'y'),
            .1,
            nu,
            rho,
            flow,
            geometry,
        )

        # intake
        for i in arrange(1.05, 1.95, .05):
            self.__rim.add((0, i), u=flow, v=0)

        # border
        for i in arrange(.05, 9.95, .05):
            self.__rim.add((i, 0), u=0, v=0)
            self.__rim.add((i, 2), u=0, v=0)

        # corner
        for i in arrange(.05, .95, .1):
            for j in arrange(.05, .95, .1):
                self.__rim.add((i, j), u=0, v=0)
        self.__rim.add((1, 1), u=0, v=0)
        for i in arrange(.05, .95, .05):
            self.__rim.add((i, 1), u=0, v=0)
        for i in arrange(.05, .95, .05):
            self.__rim.add((1, i), u=0, v=0)

        # training
        for c in self.__pde[:, :]:
            if c not in self.__rim:
                self.__pde.add(c)

        if foam or supervised:
            self.foam_grid, self.foam_cloud = self.init_foam()

        if supervised:
            self.__rim.clear()
            for k, _ in self.__pde:
                v = self.foam_cloud[k]
                self.__rim.add(k, u=v.u, v=v.v)

    @property
    def nu(self):
        return self.__nu

    @property
    def rho(self):
        return self.__rho
