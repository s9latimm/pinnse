from __future__ import annotations

import numpy as np

import src.nse.config as config
from src.base.data import Grid
from src.foam import get_foam
from src.nse.data import NSECloud


class NSEGeometry:

    @staticmethod
    def init_foam():
        u, v, p = get_foam()
        u = np.flip(u, 0).transpose().flatten()
        v = np.flip(v, 0).transpose().flatten()
        p = np.flip(p, 0).transpose().flatten()

        dim_x, dim_y = config.GEOMETRY
        num_x, num_y = config.GRID

        grid = Grid(Grid.axis(0, dim_x, num_x), Grid.axis(0, dim_y, num_y))
        cloud = NSECloud()

        for i, c in enumerate(grid):
            cloud.add(c, u=u[i], v=v[i], p=p[i])

        return grid, cloud

    def __init__(self, nu: float, rho: float, intake: float, foam: bool):
        self.nu = nu
        self.rho = rho

        if foam:
            self.foam_grid, self.foam_cloud = self.init_foam()

        dim_x, dim_y = config.GEOMETRY
        num_x, num_y = config.GRID

        self.default_grid = Grid(Grid.axis(0, dim_x, num_x), Grid.axis(0, dim_y, num_y))

        self.hires_grid = Grid(Grid.axis(0, dim_x, num_x * config.HIRES), Grid.axis(0, dim_y, num_y * config.HIRES))

        self.rimmed_grid = Grid(Grid.axis(0, dim_x, num_x, True, [.95]), Grid.axis(0, dim_y, num_y, True, [.95]))

        self.pde_cloud = NSECloud()
        self.rim_cloud = NSECloud()

        step = int(np.floor(self.rimmed_grid.height / 2))

        # intake
        for c in self.rimmed_grid[:2, step + 1:-2]:
            self.rim_cloud.add(c, u=intake, v=0)

        # upper border
        for c in self.rimmed_grid[1:-2, -2:]:
            self.rim_cloud.add(c, u=0, v=0)

        # lower border
        for c in self.rimmed_grid[1:-2, :2]:
            self.rim_cloud.add(c, u=0, v=0)

        # corner
        for c in self.rimmed_grid[1:step + 1, 2:step + 1]:
            self.rim_cloud.add(c, u=0, v=0)

        # training
        for c in self.default_grid[:, :]:
            if c not in self.rim_cloud:
                self.pde_cloud.add(c)
