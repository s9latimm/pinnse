from src.base.model.function import Parabola
from src.base.model.mesh import arrange, Grid, Axis
from src.base.model.shape import Rectangle, Figure, Line
from src.nse.model.experiments.experiment import Experiment
from src.nse.model.experiments.foam import Foam


class Block(Experiment):

    def __init__(
        self,
        nu: float = 1,
        rho: float = 1,
        flow: float = 1,
        _: bool = False,
    ) -> None:
        step = .1
        xs = Axis('x', 0, 10)
        ys = Axis('y', 0, 2)
        boundary = Figure(Line((0, 0), (10, 0)), Line((0, 2), (10, 2)))
        obstruction = Figure(Rectangle((4.7, .7), (5.3, 1.3)))

        foam = Foam(
            'block',
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
            Block.__name__,
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

        # block
        for x in arrange(4.7, 5.3, stride):
            self._knowledge.emplace((x, .7), u=0, v=0)
            self._knowledge.emplace((x, 1.3), u=0, v=0)
        for y in arrange(.7 + stride, 1.3 - stride, stride):
            self._knowledge.emplace((4.7, y), u=0, v=0)
            self._knowledge.emplace((5.3, y), u=0, v=0)

        # training
        grid = Grid(self.x.arrange(step), self.y.arrange(step))
        for c in grid:
            if self.x.start < c.x and self.y.start < c.y < self.y.stop:
                if c not in self._knowledge and c not in self.obstruction:
                    self._learning.emplace(c)
