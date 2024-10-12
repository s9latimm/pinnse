import numpy as np

import src.navier_stokes.config as config
from src.foam import get_foam


class NavierStokesGeometry:

    def __init__(self, nu, rho, intake):
        self.nu = nu
        self.rho = rho

        u, v, p = get_foam()

        dim_x, dim_y = config.GEOMETRY
        num_x, num_y = config.GRID
        half = int(num_y / 2)

        self.grid = np.dstack(
            np.meshgrid(np.linspace(0, dim_x, num_x, endpoint=False),
                        np.linspace(0, dim_y, num_y, endpoint=False),
                        indexing='ij'))

        self.mesh = np.dstack(
            np.meshgrid(np.linspace(0, dim_x, num_x * config.HIRES, endpoint=False),
                        np.linspace(0, dim_y, num_y * config.HIRES, endpoint=False),
                        indexing='ij'))

        # TODO: Clean up

        self.u = np.flip(u, 0).transpose()
        self.v = np.flip(v, 0).transpose()
        self.p = np.flip(p, 0).transpose()

        self.foam = np.hstack([
            self.grid[:, :, 0].flatten()[:, None],
            self.grid[:, :, 1].flatten()[:, None],
            self.u.flatten()[:, None],
            self.v.flatten()[:, None],
            self.p.flatten()[:, None],
        ])

        self.all = np.hstack([
            self.grid[:, :, 0].flatten()[:, None],
            self.grid[:, :, 1].flatten()[:, None],
            np.full(num_x * num_y, np.nan)[:, None],
            np.full(num_x * num_y, np.nan)[:, None],
            np.full(num_x * num_y, np.nan)[:, None],
        ])

        self.intake = np.hstack([
            self.grid[0, half:-1, 0][:, None],
            self.grid[0, half:-1, 1][:, None],
            np.full(half - 1, intake)[:, None],
            np.full(half - 1, 0)[:, None],
            np.full(half - 1, np.nan)[:, None],
        ])

        self.border = np.vstack([
            np.hstack([
                self.grid[:, 0, 0][:, None],
                self.grid[:, 0, 1][:, None],
                np.zeros(num_x)[:, None],
                np.zeros(num_x)[:, None],
                np.full(num_x, np.nan)[:, None],
            ]),
            np.hstack([
                self.grid[:, -1, 0][:, None],
                self.grid[:, -1, 1][:, None],
                np.zeros(num_x)[:, None],
                np.zeros(num_x)[:, None],
                np.full(num_x, np.nan)[:, None],
            ]),
        ])

        self.border = np.vstack([
            self.border,
            np.hstack([
                self.grid[0, :half, 0][:, None],
                self.grid[0, :half, 1][:, None],
                np.zeros(half)[:, None],
                np.zeros(half)[:, None],
                np.full(half, np.nan)[:, None],
            ]),
        ])
        for i in range(1, 10):
            self.border = np.vstack([
                self.border,
                np.hstack([
                    self.grid[i, :half, 0][:, None],
                    self.grid[i, :half, 1][:, None],
                    np.zeros(half)[:, None],
                    np.zeros(half)[:, None],
                    np.full(half, np.nan)[:, None],
                ]),
            ])

        self.exclude = np.hstack([
            self.grid[0, :half - 1, 0][:, None],
            self.grid[0, :half - 1, 1][:, None],
            np.full(half - 1, np.nan)[:, None],
            np.full(half - 1, np.nan)[:, None],
            np.full(half - 1, np.nan)[:, None],
        ])
        for i in range(1, 9):
            self.exclude = np.vstack([
                self.exclude,
                np.hstack([
                    self.grid[i, :half - 1, 0][:, None],
                    self.grid[i, :half - 1, 1][:, None],
                    np.full(half - 1, np.nan)[:, None],
                    np.full(half - 1, np.nan)[:, None],
                    np.full(half - 1, np.nan)[:, None],
                ]),
            ])

        self.geometry = np.vstack([
            self.intake,
            self.border,
        ])

        self.train = self.__invert(self.all, self.exclude)

    @staticmethod
    def __invert(ctx, itm):
        delete = []
        for i, a in enumerate(ctx):
            for b in itm:
                if a[0] == b[0] and a[1] == b[1]:
                    delete.append(i)
        return np.delete(ctx, delete, axis=0)
