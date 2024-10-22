import logging
import re
from pathlib import Path

import numpy as np

from src import OUTPUT_DIR, FOAM_DIR
from src.base.model.algebra import Real
from src.base.model.function import Null
from src.base.model.mesh import Grid, Axis, Coordinate, arrange
from src.base.model.shape import Figure, Rectangle
from src.base.view.plot import plot_seismic
from src.nse.model.experiments.experiment import Experiment
from src.nse.model.record import Record


class Foam(Experiment):
    __REGEX = r'([-+]?\d*\.?\d+[eE]?[+\-]?\d*)'

    def __init__(
        self,
        name: str,
        x: Axis,
        y: Axis,
        step: float,
        boundary: Figure,
        obstruction: Figure,
        nu: float,
        rho: float,
        flow: float,
    ) -> None:
        uid = f'{name.lower()}__{Real(step)}__{Real(nu)}__{Real(flow)}'
        self.__grid = Grid(
            x.arrange(step, True),
            y.arrange(step, True),
        )
        self.__step = step
        super().__init__(
            uid,
            x,
            y,
            boundary,
            obstruction,
            nu,
            rho,
            Null(),
            self,
        )

        path = FOAM_DIR / self._name
        if path.exists():
            self.__u, self.__v, self.__p = [], [], []
            self.__blocks = []

            self.__n = self.__dir(path)

            self.__parse_velocity(path / f'{self.__n}/U')
            self.__parse_pressure(path / f'{self.__n}/p')
            self.__parse_blocks(path / 'system' / 'blockMeshDict')

            self.__blockify()

        else:
            logging.error(f'ERROR: foam/{uid} not found')

    @property
    def grid(self) -> Grid:
        return self.__grid

    @staticmethod
    def __parse_statement(text: str, keyword: str) -> str:
        statement = re.search(keyword + r'.*?;', text, flags=re.DOTALL).group()
        return re.search(r'\(.*\)', statement, flags=re.DOTALL).group()

    def __parse_blocks(self, path: Path) -> None:
        text = path.read_text()

        statement = self.__parse_statement(text, 'vertices')
        token = re.findall(r'\(\s*([-+]?\d*\.*\d+)\s+([-+]?\d*\.*\d+)\s+([-+]?\d*\.*\d+)\s*\)', statement)

        vertices = []
        for x, y, _ in token:
            vertices.append((0, 1) + Coordinate(float(x), float(y)))

        statement = self.__parse_statement(text, 'blocks')
        token = re.findall(r'\(\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\)', statement)

        for a, _, b, _, *_ in token:
            self.__blocks.append(Rectangle(vertices[int(a)], vertices[int(b)]))

    def __parse_velocity(self, path: Path) -> None:
        text = path.read_text()

        statement = self.__parse_statement(text, 'internalField')
        token = re.findall(r'\(\s*' + self.__REGEX + r'\s+' + self.__REGEX + r'\s+' + self.__REGEX + r'\s*\)',
                           statement)

        for u, v, _ in token:
            self.__u.append(float(u))
            self.__v.append(float(v))

    def __parse_pressure(self, path: Path) -> None:
        text = path.read_text()

        statement = self.__parse_statement(text, 'internalField')
        token = re.findall(self.__REGEX, statement)

        for p in token:
            self.__p.append(float(p))

    def __blockify(self) -> None:
        i = 0
        for block in self.__blocks:
            a, b = block.shape
            for y in arrange(a.y, b.y, self.__step, True):
                for x in arrange(a.x, b.x, self.__step, True):
                    self._knowledge.insert((x, y), Record(self.__u[i], self.__v[i], self.__p[i]))
                    i += 1

        for c in self.__grid:
            if c not in self._knowledge:
                self._knowledge.insert(c, Record(np.nan, np.nan, np.nan))

    @staticmethod
    def __dir(path: Path) -> int | None:
        indices = []
        for item in path.iterdir():
            if item.is_dir() and item.name.isdigit():
                indices.append(int(item.name))
        if len(indices) > 0:
            return max(indices)
        return None


if __name__ == '__main__':

    def main():
        step = .1
        nu = 0.02
        flow = 1

        xs = Axis('x', 0, 10)
        ys = Axis('y', 0, 2)
        m = Grid(xs.arrange(step, True), ys.arrange(step, True))

        # pylint: disable=import-outside-toplevel,cyclic-import
        from src.nse.model.experiments import EXPERIMENTS
        for experiment in EXPERIMENTS:
            f = Foam(
                experiment,
                xs,
                ys,
                step,
                Figure(),
                Figure(),
                nu,
                1,
                flow,
            )
            d = m.transform(f.knowledge)

            # f.knowledge.save(OUTPUT_DIR / 'foam' / f'{experiment}_uvp.csv')

            print(len(f.knowledge))

            plot_seismic(
                f.name,
                m.x,
                m.y,
                [
                    ('u', d.u),
                    ('v', d.v),
                    ('p', d.p - np.nanmin(d.p)),
                ],
                path=OUTPUT_DIR / 'foam' / f'{experiment}_uvp.pdf',
                boundary=f.boundary,
                figure=f.obstruction,
            )

            # plot_stream(
            #     f.name,
            #     m.x,
            #     m.y,
            #     d.u,
            #     d.v,
            #     path=OUTPUT_DIR / 'foam' / f'{experiment}_str.pdf',
            #     boundary=f.boundary,
            #     figure=f.obstruction,
            # )
            #
            # plot_arrows(
            #     f.name,
            #     m.x,
            #     m.y,
            #     d.u,
            #     d.v,
            #     path=OUTPUT_DIR / 'foam' / f'{experiment}_arw.pdf',
            #     boundary=f.boundary,
            #     figure=f.obstruction,
            # )

    main()
