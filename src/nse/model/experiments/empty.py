from src.base.model.function import Parabola
from src.base.model.mesh import arrange, Grid, Axis
from src.base.model.shape import Figure, Line
from src.nse.model.experiments.experiment import Experiment
from src.nse.model.experiments.foam import Foam


class Empty(Experiment):

    def __init__(
        self,
        nu: float = 1,
        rho: float = 1,
        flow: float = 1,
        _: bool = False,
    ) -> None:
        name = Empty.__name__
        step = .1
        xs = Axis('x', 0, 10)
        ys = Axis('y', 0, 2)
        boundary = Figure(Line((0, 0), (10, 0)), Line((0, 2), (10, 2)))
        obstruction = Figure()

        foam = Foam(
            name,
            xs,
            ys,
            step,
            boundary,
            obstruction,
            nu,
            rho,
            flow,
        )

        super().__init__(
            name,
            xs,
            ys,
            boundary,
            obstruction,
            nu,
            rho,
            Parabola(0, 2, flow),
            foam,
        )

        stride = step / 2

        # inlet
        for y in arrange(0, 2, stride):
            self._inlet.emplace((0, y), u=self._in(y), v=0)
            self._outlet.emplace((10, y))

        # border
        for x in arrange(stride, 10, stride):
            self._knowledge.emplace((x, 0), u=0, v=0)
            self._knowledge.emplace((x, 2), u=0, v=0)

        # training
        grid = Grid(self.x.arrange(step), self.y.arrange(step))
        for c in grid:
            if self.x.start < c.x and self.y.start < c.y < self.y.stop:
                if c not in self._knowledge and c not in self.obstruction:
                    self._learning.emplace(c)
