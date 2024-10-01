from __future__ import annotations

import typing as t

import numpy as np

import src.nse.config as config
from src.base.data import Grid, arrange
from src.foam import get_foam
from src.nse.data import NSECloud
from src.nse.experiments.step import Step


class NSEGeometry:

    @staticmethod
    def init_foam() -> t.Tuple[Grid, NSECloud]:
        u, v, p = get_foam()
        u = np.flip(u, 0).transpose().flatten()
        v = np.flip(v, 0).transpose().flatten()
        p = np.flip(p, 0).transpose().flatten()

        dim_x, dim_y = config.GEOMETRY
        num_x, num_y = config.GRID

        grid = Grid(
            arrange(.05, 9.95, .1),
            arrange(.05, 1.95, .1),
        )
        cloud = NSECloud()

        for i, c in enumerate(grid):
            cloud.add(c, u=u[i], v=v[i], p=p[i])

        return grid, cloud

    def __init__(self, nu: float, rho: float, intake: float, foam: bool = False, supervised: bool = False) -> None:

        experiment = Step(nu, rho, intake)

        self.pde_grid = experiment.finite_grid

        self.hires_grid = experiment.hires_grid

        self.rim_grid = experiment.layout_grid

        self.pde_cloud = NSECloud()
        self.rim_cloud = NSECloud()

        if foam or supervised:
            self.foam_grid, self.foam_cloud = self.init_foam()

        if supervised:
            self.rim_cloud = NSECloud()
            for k, _ in self.pde_cloud:
                v = self.foam_cloud[k]
                self.rim_cloud.add(k, u=v.u, v=v.v)
