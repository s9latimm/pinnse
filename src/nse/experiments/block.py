from src import FOAM_DIR
from src.base.mesh import arrange, Mesh, Axis
from src.base.shape import Rectangle, Figure
from src.nse.experiments.experiment import NSEExperiment, inlet
from src.nse.experiments.foam import Foam


class Block(NSEExperiment):

    def __init__(
        self,
        nu: float,
        rho: float,
        flow: float,
        _: bool,
    ) -> None:
        mesh = Mesh(Axis('x', 0, 10).arrange(.01, True), Axis('y', 0, 2).arrange(.01, True))
        foam = Foam(
            FOAM_DIR / 'block_01',
            mesh,
            [(0, 1.5, 1, 2), (0, 0.5, 1, 1.5), (0, 0, 1, 0.5), (1, 1.5, 2, 2), (1, 0, 2, 0.5), (2, 1.5, 10, 2),
             (2, 0.5, 10, 1.5), (2, 0, 10, 0.5)],
            100,
            Figure(Rectangle((0, 0), (10, 2))),
            Figure(Rectangle((0, 0), (1, 1))),
            0.01,
            1.,
        )
        super().__init__(
            'Step',
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            Figure(Rectangle((0, 0), (10, 2))),
            Figure(Rectangle((2, .5), (3, 1.5))),
            nu,
            rho,
            flow,
            foam,
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

        for figure in self.obstruction:
            for c in figure[::s]:
                self._knowledge.emplace(c, u=0, v=0)

        # training
        mesh = Mesh(self.x.arrange(t), self.y.arrange(t))
        for c in mesh:
            if c not in self._knowledge and c not in self._learning and c not in self.obstruction:
                self._learning.emplace(c)
