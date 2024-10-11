from src import FOAM_DIR
from src.base.mesh import arrange, Mesh, Axis
from src.base.shape import Rectangle, Figure
from src.nse.experiments.experiment import NSEExperiment, inlet
from src.nse.experiments.foam import Foam


class Step(NSEExperiment):

    def __init__(
        self,
        nu: float,
        rho: float,
        flow: float,
        supervised: bool,
    ) -> None:
        mesh = Mesh(Axis('x', 0, 10).arrange(.1, True), Axis('y', 0, 2).arrange(.1, True))
        foam = Foam(
            FOAM_DIR / 'step_01',
            mesh,
            [(0, 0, 1, 1), (1, 1, 10, 2), (1, 0, 10, 1)],
            10,
            Figure(Rectangle((0, 0), (10, 2))),
            Figure(Rectangle((0, 0), (1, 1))),
            0.08,
            1.,
        )
        super().__init__(
            'Step',
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            Figure(Rectangle((0, 0), (10, 2))),
            Figure(Rectangle((0, 0), (1, 1))),
            nu,
            rho,
            flow,
            foam,
            supervised,
        )

        t = .1
        s = t / 2

        # inlet
        for y in arrange(1, 2, s):
            u = inlet(1, 2, flow)(y)
            self._knowledge.add((0, y), u=u, v=0)
            self._outlet.add((10, y))

        # border
        for x in arrange(1, 10, s):
            self._knowledge.add((x, 0), u=0, v=0)
        for x in arrange(s, 10, s):
            self._knowledge.add((x, 2), u=0, v=0)

        for x in arrange(s, 1, s):
            self._knowledge.add((x, 1), u=0, v=0)
        for y in arrange(s, 1 - s, s):
            self._knowledge.add((1, y), u=0, v=0)

        # training
        mesh = Mesh(self.x.arrange(t), self.y.arrange(t))
        for c in mesh:
            if c not in self._knowledge and c not in self._learning and c not in self.obstruction:
                self._learning.add(c)

        if supervised:
            self._knowledge = self.foam.knowledge
