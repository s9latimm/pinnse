from src.base.geometry import arrange, Mesh
from src.base.shapes import Rectangle
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
        step = Rectangle((0, 0), (1, 1))

        super().__init__(
            'Step',
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            nu,
            rho,
            inlet,
            foam,
            supervised,
            [step, Rectangle((0, 0), (10, 2))],
        )

        # inlet
        for y in arrange(1, 2, .05):
            u = inlet * (1. - (3. - 2. * y)**2)
            self._knowledge.add((0, y), u=u, v=0)

        # border
        for x in arrange(1, 10, .05):
            self._knowledge.add((x, 0), u=0, v=0)
        for x in arrange(.05, 10, .05):
            self._knowledge.add((x, 2), u=0, v=0)

        # obstruction
        # for x in arrange(.05, .95, .1):
        #     for y in arrange(.05, .95, .1):
        #         self._knowledge.add((x, y), u=0, v=0)
        for x in arrange(.05, 1, .05):
            self._knowledge.add((x, 1), u=0, v=0)
        for y in arrange(.05, .95, .05):
            self._knowledge.add((1, y), u=0, v=0)

        # training
        mesh = Mesh(self.x.arrange(.1, True), self.y.arrange(.1, True))
        for c in mesh:
            if c not in self._knowledge and c not in step:
                self._learning.add(c)

        # if supervised:
        #     self._knowledge.clear()
        #     for k, _ in self._knowledge:
        #         v = self._foam_facts[k]
        #         self._knowledge.add(k, u=v.u, v=v.v)
