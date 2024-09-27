import numpy as np

import src.navier_stokes.config as config
from src.foam import get_foam


def disc(start, stop, num, padding=(False, False), extra=()):
    s = np.linspace(start, stop, num, endpoint=False)
    p = (stop - start) / num / 2
    if padding[1]:
        s = np.concatenate((s, [stop - p, stop]))
    if padding[0]:
        s = np.concatenate(([start - 2 * p, start - p], s))
    return np.sort(np.concatenate((extra, s)))


class NavierStokesGeometry:

    def import_foam(self):
        u, v, p = get_foam()

        dim_x, dim_y = config.GEOMETRY
        num_x, num_y = config.GRID

        self.i_grid = np.dstack(
            np.meshgrid(np.linspace(0, dim_x, num_x, endpoint=False),
                        np.linspace(0, dim_y, num_y, endpoint=False),
                        indexing='ij'))

        return np.hstack([
            self.i_grid[:, :, 0].flatten()[:, None],
            self.i_grid[:, :, 1].flatten()[:, None],
            np.flip(u, 0).transpose().flatten()[:, None],
            np.flip(v, 0).transpose().flatten()[:, None],
            np.flip(p, 0).transpose().flatten()[:, None],
        ])

    def __init__(self, nu, rho, intake):
        self.nu = nu
        self.rho = rho

        self.foam = self.import_foam()

        dim_x, dim_y = config.GEOMETRY
        i_x, i_y = config.GRID

        self.i_grid = np.dstack(np.meshgrid(disc(0, dim_x, i_x), disc(0, dim_y, i_y), indexing='ij'))
        self.i_stack = np.hstack([
            self.i_grid[:, :, 0].flatten()[:, None],
            self.i_grid[:, :, 1].flatten()[:, None],
            np.full(i_x * i_y, np.nan)[:, None],
            np.full(i_x * i_y, np.nan)[:, None],
            np.full(i_x * i_y, np.nan)[:, None],
        ])

        self.h_grid = np.dstack(
            np.meshgrid(disc(0, dim_x, i_x * config.HIRES), disc(0, dim_y, i_y * config.HIRES), indexing='ij'))
        self.h_stack = np.hstack([
            self.h_grid[:, :, 0].flatten()[:, None],
            self.h_grid[:, :, 1].flatten()[:, None],
            np.full(i_x * i_y * config.HIRES**2, np.nan)[:, None],
            np.full(i_x * i_y * config.HIRES**2, np.nan)[:, None],
            np.full(i_x * i_y * config.HIRES**2, np.nan)[:, None],
        ])

        o_x, o_y = i_x + 5, i_y + 5
        o_h = int(np.floor(o_y / 2))
        self.o_grid = np.dstack(
            np.meshgrid(disc(0, dim_x, i_x, (True, True), [.95]),
                        disc(0, dim_y, i_y, (True, True), [.95]),
                        indexing='ij'))
        self.o_stack = np.hstack([
            self.o_grid[:, :, 0].flatten()[:, None],
            self.o_grid[:, :, 1].flatten()[:, None],
            np.full(o_x * o_y, np.nan)[:, None],
            np.full(o_x * o_y, np.nan)[:, None],
            np.full(o_x * o_y, np.nan)[:, None],
        ])

        self.intake = np.hstack([
            self.o_grid[:2, o_h + 1:-2, 0].flatten()[:, None],
            self.o_grid[:2, o_h + 1:-2, 1].flatten()[:, None],
            np.full((o_h - 2) * 2, intake)[:, None],
            np.full((o_h - 2) * 2, 0)[:, None],
            np.full((o_h - 2) * 2, np.nan)[:, None],
        ])

        self.b_stack = np.vstack([
            self.intake,
            np.hstack([
                self.o_grid[:, :2, 0].flatten()[:, None],
                self.o_grid[:, :2, 1].flatten()[:, None],
                np.zeros(o_x * 2)[:, None],
                np.zeros(o_x * 2)[:, None],
                np.full(o_x * 2, np.nan)[:, None],
            ]),
            np.hstack([
                self.o_grid[:, -2:, 0].flatten()[:, None],
                self.o_grid[:, -2:, 1].flatten()[:, None],
                np.zeros(o_x * 2)[:, None],
                np.zeros(o_x * 2)[:, None],
                np.full(o_x * 2, np.nan)[:, None],
            ]),
        ])

        print(o_h)

        self.b_stack = np.vstack([
            self.b_stack,
            # np.hstack([
            #     self.o_grid[0, :o_h, 0][:, None],
            #     self.o_grid[0, :o_h, 1][:, None],
            #     np.zeros(o_h)[:, None],
            #     np.zeros(o_h)[:, None],
            #     np.full(o_h, np.nan)[:, None],
            # ]),
            np.hstack([
                self.o_grid[1:o_h + 1, :o_h + 1, 0].flatten()[:, None],
                self.o_grid[1:o_h + 1, :o_h + 1, 1].flatten()[:, None],
                np.zeros(o_h * (o_h + 1))[:, None],
                np.zeros(o_h * (o_h + 1))[:, None],
                np.full(o_h * (o_h + 1), np.nan)[:, None],
            ]),
        ])

        self.exclude = np.vstack([
            np.hstack([
                self.o_grid[:1, -2:, 0].flatten()[:, None],
                self.o_grid[:1, -2:, 1].flatten()[:, None],
                np.full(2, np.nan)[:, None],
                np.full(2, np.nan)[:, None],
                np.full(2, np.nan)[:, None],
            ]),
            np.hstack([
                self.o_grid[:1, :o_h + 1, 0].flatten()[:, None],
                self.o_grid[:1, :o_h + 1, 1].flatten()[:, None],
                np.full(o_h + 1, np.nan)[:, None],
                np.full(o_h + 1, np.nan)[:, None],
                np.full(o_h + 1, np.nan)[:, None],
            ]),
            # np.hstack([
            #     self.o_grid[0, o_h - 1, 0].flatten()[:, None],
            #     self.o_grid[0, o_h - 1, 1].flatten()[:, None],
            #     np.full(1, np.nan)[:, None],
            #     np.full(1, np.nan)[:, None],
            #     np.full(1, np.nan)[:, None],
            # ]),
            # np.hstack([
            #     self.o_grid[0:11, :o_h, 0].flatten()[:, None],
            #     self.o_grid[0:11, :o_h, 1].flatten()[:, None],
            #     np.full(o_h**2, np.nan)[:, None],
            #     np.full(o_h**2, np.nan)[:, None],
            #     np.full(o_h**2, np.nan)[:, None],
            # ]),
            np.hstack([
                self.o_grid[:o_h - 1, 0, 0][:, None],
                self.o_grid[:o_h - 1, 0, 1][:, None],
                np.full(o_h - 1, np.nan)[:, None],
                np.full(o_h - 1, np.nan)[:, None],
                np.full(o_h - 1, np.nan)[:, None],
            ]),
            np.hstack([
                self.o_grid[o_h, 0, 0],
                self.o_grid[o_h, 0, 1],
                np.full(1, np.nan),
                np.full(1, np.nan),
                np.full(1, np.nan),
            ]),
            np.hstack([
                self.o_grid[:o_h, 1, 0][:, None],
                self.o_grid[:o_h, 1, 1][:, None],
                np.full(o_h, np.nan)[:, None],
                np.full(o_h, np.nan)[:, None],
                np.full(o_h, np.nan)[:, None],
            ]),
            np.hstack([
                self.o_grid[o_h + 1:, o_h, 0][:, None],
                self.o_grid[o_h + 1:, o_h, 1][:, None],
                np.full(o_x - o_h - 1, np.nan)[:, None],
                np.full(o_x - o_h - 1, np.nan)[:, None],
                np.full(o_x - o_h - 1, np.nan)[:, None],
            ]),
            np.hstack([
                self.o_grid[o_h, o_h + 1:, 0][:, None],
                self.o_grid[o_h, o_h + 1:, 1][:, None],
                np.full(o_h, np.nan)[:, None],
                np.full(o_h, np.nan)[:, None],
                np.full(o_h, np.nan)[:, None],
            ]),
            np.hstack([
                self.o_grid[-2:, :, 0].flatten()[:, None],
                self.o_grid[-2:, :, 1].flatten()[:, None],
                np.full(o_y * 2, np.nan)[:, None],
                np.full(o_y * 2, np.nan)[:, None],
                np.full(o_y * 2, np.nan)[:, None],
            ]),
        ])

        self.g_stack = np.vstack([
            self.intake,
            self.b_stack,
        ])

        self.t_stack = self.__invert(self.o_stack, self.exclude)

        self.b_stack = self.__invert(self.b_stack, self.__invert(self.o_stack, self.t_stack))

    @staticmethod
    def __invert(ctx, itm):
        delete = []
        for i, a in enumerate(ctx):
            for b in itm:
                if a[0] == b[0] and a[1] == b[1]:
                    delete.append(i)
        return np.delete(ctx, delete, axis=0)
