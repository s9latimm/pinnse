from src.base.mesh import arrange, Mesh, Axis
from src.base.shape import Rectangle, Figure
from src.nse.experiments.experiment import NSEExperiment, inlet


class Empty(NSEExperiment):

    def __init__(
        self,
        nu: float,
        rho: float,
        flow: float,
        _: bool,
    ) -> None:
        super().__init__(
            'Step',
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            Figure(Rectangle((0, 0), (10, 2))),
            Figure(),
            nu,
            rho,
            flow,
        )

        t = .1
        s = t / 2

        # inlet
        for y in arrange(0, 2, s):
            u = inlet(0, 2, flow)(y)
            self._knowledge.emplace((0, y), u=u, v=0)
            self._outlet.emplace((10, y))

        # border
        for x in arrange(s, 10 - s, s):
            self._knowledge.emplace((x, 0), u=0, v=0)
            self._knowledge.emplace((x, 2), u=0, v=0)

        # training
        mesh = Mesh(self.x.arrange(t), self.y.arrange(t))
        for c in mesh:
            if c not in self._knowledge and c not in self._learning:
                self._learning.emplace(c)
