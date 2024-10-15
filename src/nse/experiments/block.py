from src import FOAM_DIR
from src.base.function import Parabola
from src.base.mesh import arrange, Grid, Axis
from src.base.shape import Rectangle, Figure, Line
from src.nse.experiments.experiment import Experiment
from src.nse.experiments.foam import Foam


class Block(Experiment):

    def __init__(
        self,
        nu: float = 1,
        rho: float = 1,
        flow: float = 1,
        _: bool = False,
    ) -> None:
        grid = Grid(Axis('x', 0, 10).arrange(.01, True), Axis('y', 0, 2).arrange(.01, True))
        foam = Foam(
            FOAM_DIR / 'block_01',
            grid,
            [(0, 1.5, 1, 2), (0, 0.5, 1, 1.5), (0, 0, 1, 0.5), (1, 1.5, 2, 2), (1, 0, 2, 0.5), (2, 1.5, 10, 2),
             (2, 0.5, 10, 1.5), (2, 0, 10, 0.5)],
            100,
            Figure(Line((0, 0), (10, 0)), Line((0, 2), (10, 2))),
            Figure(Rectangle((0, 0), (1, 1))),
            0.01,
            1.,
        )
        super().__init__(
            Block.__name__,
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            Figure(Line((0, 0), (10, 0)), Line((0, 2), (10, 2))),
            Figure(Rectangle((4.7, .7), (5.3, 1.3))),
            nu,
            rho,
            Parabola(0, 2, flow),
            foam,
        )

        t = .1
        s = t / 2

        # inlet
        for y in arrange(0, 2, s):
            self._inlet.emplace((0, y), u=self._in(y), v=0)
            self._outlet.emplace((10, y))

        # border
        for x in arrange(s, 10 - s, s):
            self._knowledge.emplace((x, 0), u=0, v=0)
            self._knowledge.emplace((x, 2), u=0, v=0)

        for figure in self.obstruction:
            for c in figure[::s]:
                self._knowledge.emplace(c, u=0, v=0)

        # training
        grid = Grid(self.x.arrange(t), self.y.arrange(t))
        for c in grid:
            if c not in self._knowledge and c not in self._learning and c not in self.obstruction:
                self._learning.emplace(c)
