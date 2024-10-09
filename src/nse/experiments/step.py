from src.base.mesh import arrange, Mesh
from src.base.shape import Rectangle, Figure
from src.nse.experiments.experiment import Axis, NSEExperiment


class Step(NSEExperiment):

    def __init__(
        self,
        nu: float,
        rho: float,
        inlet: float,
        foam: bool,
        supervised: bool,
    ) -> None:
        super().__init__(
            'Step',
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            nu,
            rho,
            inlet,
            foam,
            supervised,
            Figure(Rectangle((0, 0), (10, 2))),
            Figure(Rectangle((0, 0), (1, 1))),
        )

        s = 1. / 19
        t = 1. / 11

        # inlet
        for y in arrange(1, 2, s):
            u = inlet * (1. - (3. - 2. * y)**2)
            self._knowledge.add((0, y), u=u, v=0)

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
        mesh = Mesh(self.x.arrange(t, True), self.y.arrange(t, True))
        for c in mesh:
            if c not in self._knowledge and c not in self._learning and c not in self.obstruction:
                self._learning.add(c)

        # if supervised:
        #     self._knowledge.clear()
        #     for k, _ in self._knowledge:
        #         v = self._foam_facts[k]
        #         self._knowledge.add(k, u=v.u, v=v.v)
