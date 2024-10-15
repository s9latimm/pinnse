from src.base.function import Parabola
from src.base.mesh import arrange, Grid, Axis
from src.base.shape import Figure, Line
from src.nse.experiments.experiment import NSEExperiment


class Empty(NSEExperiment):

    def __init__(
        self,
        nu: float = 1,
        rho: float = 1,
        flow: float = 1,
        _: bool = False,
    ) -> None:
        super().__init__(
            Empty.__name__,
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            Figure(Line((0, 0), (10, 0)), Line((0, 2), (10, 2))),
            Figure(),
            nu,
            rho,
            Parabola(0, 2, flow),
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

        # training
        grid = Grid(self.x.arrange(t), self.y.arrange(t))
        for c in grid:
            if c not in self._knowledge and c not in self._learning:
                self._learning.emplace(c)
