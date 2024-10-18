from src import FOAM_DIR
from src.base.model.function import Parabola
from src.base.model.mesh import arrange, Grid, Axis
from src.base.model.shape import Rectangle, Figure, Line
from src.nse.model.experiments.experiment import Experiment
from src.nse.model.experiments.foam import Foam


class Slalom(Experiment):

    def __init__(
        self,
        nu: float = 1,
        rho: float = 1,
        flow: float = 1,
        supervised: bool = False,
    ) -> None:
        grid = Grid(Axis('x', 0, 10).arrange(.1, True), Axis('y', 0, 2).arrange(.1, True))
        foam = Foam(
            FOAM_DIR / 'slalom',
            grid,
            .1,
            Figure(Line((0, 0), (10, 0)), Line((0, 2), (10, 2))),
            Figure(Rectangle((0, 0), (1, 1))),
            0.08,
            1.,
        )
        super().__init__(
            Slalom.__name__,
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            Figure(Line((0, 0), (10, 0)), Line((0, 2), (10, 2))),
            Figure(Rectangle((0, 0), (1, 1)), Rectangle((4.5, 1), (5.5, 2))),
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
        for x in arrange(1, 9, s):
            self._knowledge.emplace((x, 0), u=0, v=0)
        for x in arrange(s, 4.5, s):
            self._knowledge.emplace((x, 2), u=0, v=0)
        for x in arrange(5.5, 10, s):
            self._knowledge.emplace((x, 2), u=0, v=0)

        for x in arrange(s, 1, s):
            self._knowledge.emplace((x, 1), u=0, v=0)
        for y in arrange(s, 1 - s, s):
            self._knowledge.emplace((1, y), u=0, v=0)
            self._knowledge.emplace((9, y), u=0, v=0)

        for x in arrange(4.5, 5.5, s):
            self._knowledge.emplace((x, 1), u=0, v=0)
        for y in arrange(1 + s, 2 - s, s):
            self._knowledge.emplace((4.5, y), u=0, v=0)
            self._knowledge.emplace((5.5, y), u=0, v=0)

        for x in arrange(9, 10, s):
            self._knowledge.emplace((x, 1), u=0, v=0)

        # training
        grid = Grid(self.x.arrange(t), self.y.arrange(t))
        for c in grid:
            if c not in self._knowledge and c not in self._learning and c not in self.obstruction:
                self._learning.emplace(c)

        if supervised:
            self._knowledge = self.foam.knowledge
