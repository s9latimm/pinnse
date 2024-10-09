from src.base.mesh import arrange, Mesh
from src.base.shape import Rectangle, Figure
from src.nse.experiments.experiment import Axis, NSEExperiment


class Block(NSEExperiment):

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
            Figure(Rectangle((1, .5), (2, 1.5))),
        )

        # intake
        for i in arrange(.05, 1.95, .05):
            self._knowledge.add((0, i), u=inlet, v=0)

        # border
        for i in arrange(.05, 9.95, .05):
            self._knowledge.add((i, 0), u=0, v=0)
            self._knowledge.add((i, 2), u=0, v=0)

        # corner
        for i in arrange(1.05, 1.95, .1):
            for j in arrange(.55, 1.45, .1):
                self._knowledge.add((i, j), u=0, v=0)
        for i in arrange(1.05, 1.95, .05):
            self._knowledge.add((i, .5), u=0, v=0)
            self._knowledge.add((i, 1.5), u=0, v=0)
        for i in arrange(.55, 1.45, .05):
            self._knowledge.add((1, i), u=0, v=0)
            self._knowledge.add((2, i), u=0, v=0)
        self._knowledge.add((1, .5), u=0, v=0)
        self._knowledge.add((1, 1.5), u=0, v=0)
        self._knowledge.add((2, .5), u=0, v=0)
        self._knowledge.add((2, 1.5), u=0, v=0)

        # training
        mesh = Mesh(self.x.arrange(.1, True), self.y.arrange(.1, True))
        for c in mesh:
            if c not in self._knowledge:
                self._learning.add(c)
