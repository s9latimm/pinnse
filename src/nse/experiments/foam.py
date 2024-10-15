import re
from pathlib import Path

import numpy as np

from src import OUTPUT_DIR, FOAM_DIR
from src.base.mesh import Grid, Axis
from src.base.plot import plot_heatmaps
from src.base.shape import Figure, Rectangle
from src.nse.experiments.experiment import NSEExperiment


class Foam(NSEExperiment):

    def __init__(
        self,
        path: Path,
        grid: Grid,
        model: list[tuple[float, float, float, float]],
        scale: float,
        boundary: Figure = None,
        obstruction: Figure = None,
        nu: float = 1.,
        rho: float = 1.,
    ):
        self.__grid = grid
        super().__init__(
            'Foam',
            self.__grid.x,
            self.__grid.y,
            boundary,
            obstruction,
            nu,
            rho,
        )

        directory = self.__dir(path)

        # Read data
        u_raw, v_raw = self.__parse_velocity(path / f'{directory}/U')
        p_raw = self.__parse_pressure(path / f'{directory}/p')

        model = self.__scale(model, scale)
        u = self.__convert(u_raw, model)
        v = self.__convert(v_raw, model)
        p = self.__convert(p_raw, model)

        u = np.flip(u, 0).transpose().flatten()
        v = np.flip(v, 0).transpose().flatten()
        p = np.flip(p, 0).transpose().flatten()

        for i, c in enumerate(self.__grid):
            self._knowledge.emplace(c, u=u[i], v=v[i], p=p[i])

    @property
    def grid(self) -> Grid:
        return self.__grid

    @staticmethod
    def __parse_velocity(path: Path) -> tuple[list[float], list[float]]:
        u, v = [], []

        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Flag to start collecting data
        munch = False

        for line in lines:
            line = line.strip()

            # Start of data section
            if line.startswith("internalField"):
                munch = True
                continue

            # End of data section
            if munch and line.startswith(";"):
                munch = False
                continue

            if munch:
                # Extract tuples using regular expression
                matches = re.findall(r'\(([^)]+)\)', line)
                for match in matches:
                    # Convert list string to list of floats
                    data = list(map(float, match.split()))
                    u.append(data[0])
                    v.append(data[1])

        return u, v

    @staticmethod
    def __parse_pressure(path: Path) -> list[float]:
        p = []

        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Flag to start collecting data
        munch = False

        for line in lines:
            line = line.strip()

            # End of data section
            if munch and line.startswith(")"):
                munch = False
                continue

            if munch:
                # match = []
                # match = re.findall(r'-?\d+\.\d+', line)
                # # Extract tuples using regular expression
                # if match != []:
                #     data.append(float(match[0]))
                p.append(float(line))

            # Start of data section
            if line.startswith("("):
                munch = True
                continue

        return p

    @staticmethod
    def __convert(data: list[float], model: list[tuple[int, int, int, int]]) -> np.ndarray:
        """
        model: a list of tuples, describes the parts of the model (start_x, start_y, end_x, end_y)
        """
        min_x = min(tup[0] for tup in model)
        min_y = min(tup[1] for tup in model)
        size_x = max(tup[2] for tup in model) - min_x
        size_y = max(tup[3] for tup in model) - min_y

        values = np.zeros((size_y, size_x))

        head = 0
        for mode in model:
            px = mode[2] - mode[0]
            py = mode[3] - mode[1]

            tail = head + px * py
            part = data[head:tail]

            for y in range(py):
                for x in range(px):
                    values[mode[1] + y - min_y][mode[0] + x - min_x] = part[(py - 1 - y) * px + x]

            head = tail

        return values

    @staticmethod
    def __dir(path: Path) -> int | None:
        indices = []
        for item in path.iterdir():
            if item.is_dir() and item.name.isdigit():
                indices.append(int(item.name))
        if len(indices) > 0:
            return max(indices)
        return None

    @staticmethod
    def __scale(model, scale) -> list[tuple[int, int, int, int]]:
        # scales every value in every tuple
        return [(int(i[0] * scale), int(i[1] * scale), int(i[2] * scale), int(i[3] * scale)) for i in model]


if __name__ == '__main__':
    m = Grid(Axis('x', 0, 10).arrange(.1, True), Axis('y', 0, 2).arrange(.1, True))
    f = Foam(
        FOAM_DIR / 'step_01',
        m,
        [(0, 0, 1, 1), (1, 1, 10, 2), (1, 0, 10, 1)],
        10,
        Figure(Rectangle((0, 0), (10, 2))),
        Figure(Rectangle((0, 0), (1, 1))),
        0.08,
        1.,
    )
    d = m.transform(f.knowledge)

    plot_heatmaps(
        'OpenFOAM',
        m.x,
        m.y,
        [
            ('u', d.u),
            ('v', d.v),
            ('p', d.p),
        ],
        path=OUTPUT_DIR / 'foam' / 'foam_uvp.pdf',
        boundary=f.boundary,
        figure=f.obstruction,
    )
