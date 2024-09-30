from __future__ import annotations

import typing as t

import numpy as np

import src.nse.config as config
from src.base.data import Grid, arrange
from src.foam import get_foam
from src.nse.data import NSECloud


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
        self.nu = nu
        self.rho = rho

        self.pde_grid = Grid(
            arrange(.05, 9.95, .1),
            arrange(.05, 1.95, .1),
        )

        self.hires_grid = Grid(
            arrange(.005, 9.995, .01),
            arrange(.005, 1.995, .01),
        )

        self.rim_grid = Grid(
            arrange(0, 10, .05),
            arrange(0, 2, .05),
        )

        self.pde_cloud = NSECloud()
        self.rim_cloud = NSECloud()

        if foam or supervised:
            self.foam_grid, self.foam_cloud = self.init_foam()

        # intake
        for i in arrange(1.05, 1.95, .05):
            self.rim_cloud.add((0, i), u=intake, v=0)

        # border
        for i in arrange(.05, 9.95, .05):
            self.rim_cloud.add((i, 0), u=0, v=0)
            self.rim_cloud.add((i, 2), u=0, v=0)

        # corner
        for i in arrange(.05, .95, .1):
            for j in arrange(.05, .95, .1):
                self.rim_cloud.add((i, j), u=0, v=0)
        self.rim_cloud.add((1, 1), u=0, v=0)
        for i in arrange(.05, .95, .05):
            self.rim_cloud.add((i, 1), u=0, v=0)
        for i in arrange(.05, .95, .05):
            self.rim_cloud.add((1, i), u=0, v=0)

        # training
        for c in self.pde_grid[:, :]:
            if c not in self.rim_cloud:
                self.pde_cloud.add(c)

        if supervised:
            self.rim_cloud = NSECloud()
            for k, _ in self.pde_cloud:
                v = self.foam_cloud[k]
                self.rim_cloud.add(k, u=v.u, v=v.v)
