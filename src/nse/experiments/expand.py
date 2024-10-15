from src import FOAM_DIR
from src.base.model.function import Parabola
from src.base.model.mesh import arrange, Grid, Axis
from src.base.model.shape import Rectangle, Figure, Line
from src.nse.experiments.experiment import Experiment
from src.nse.experiments.foam import Foam


class Expand(Experiment):

    def __init__(
        self,
        nu: float = 1,
        rho: float = 1,
        flow: float = 1,
        supervised: bool = False,
    ) -> None:
        grid = Grid(Axis('x', 0, 10).arrange(.1, True), Axis('y', 0, 2).arrange(.1, True))
        foam = Foam(
            FOAM_DIR / 'step_01',
            grid,
            [(0, 0, 1, 1), (1, 1, 10, 2), (1, 0, 10, 1)],
            10,
            Figure(Line((0, 0), (10, 0)), Line((0, 2), (10, 2))),
            Figure(Rectangle((0, 0), (1, 1))),
            0.08,
            1.,
        )
        super().__init__(
            Expand.__name__,
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            Figure(Line((0, 0), (10, 0)), Line((0, 2), (10, 2))),
            Figure(Rectangle((0, 0), (1, 1)), Rectangle((9, 0), (10, 1))),
            nu,
            rho,
            Parabola(1, 2, flow),
            foam,
            supervised,
        )

        t = .1
        s = t / 2

        # inlet
        for y in arrange(1, 2, s):
            self._inlet.emplace((0, y), u=self._in(y), v=0)
            self._outlet.emplace((10, y))

        # border
        for x in arrange(1, 10, s):
            self._knowledge.emplace((x, 0), u=0, v=0)
        for x in arrange(s, 10, s):
            self._knowledge.emplace((x, 2), u=0, v=0)

        for x in arrange(s, 1, s):
            self._knowledge.emplace((x, 1), u=0, v=0)
        for y in arrange(s, 1 - s, s):
            self._knowledge.emplace((1, y), u=0, v=0)

        # training
        grid = Grid(self.x.arrange(t), self.y.arrange(t))
        for c in grid:
            if c not in self._knowledge and c not in self._learning and c not in self.obstruction:
                self._learning.emplace(c)

        if supervised:
            self._knowledge = self.foam.knowledge
