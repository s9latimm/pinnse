from src.base.geometry import arrange, Mesh
from src.base.shapes import Line
from src.nse.experiments.experiment import Axis, NSEExperiment


class Step(NSEExperiment):

    def __init__(
        self,
        nu: float,
        rho: float,
        flow: float,
        foam: bool,
        supervised: bool,
    ):
        self.__nu = nu
        self.__rho = rho

        geometry = [
            Line((1, 0), (1, 1)),
            Line((0, 1), (1, 1)),
        ]

        super().__init__(
            'Step',
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            .1,
            nu,
            rho,
            flow,
            foam,
            supervised,
            geometry,
        )

        # intake
        for i in arrange(1.05, 1.95, .05):
            self._knowledge.add((0, i), u=flow, v=0)

        # border
        for i in arrange(.05, 9.95, .05):
            self._knowledge.add((i, 0), u=0, v=0)
            self._knowledge.add((i, 2), u=0, v=0)

        # corner
        for i in arrange(.05, .95, .1):
            for j in arrange(.05, .95, .1):
                self._knowledge.add((i, j), u=0, v=0)
        self._knowledge.add((1, 1), u=0, v=0)
        for i in arrange(.05, .95, .05):
            self._knowledge.add((i, 1), u=0, v=0)
        for i in arrange(.05, .95, .05):
            self._knowledge.add((1, i), u=0, v=0)

        # training
        mesh = Mesh(self.x.arrange(.1, True), self.y.arrange(.1, True))
        for c in mesh:
            if c not in self._knowledge:
                self._learning.add(c)

        # if supervised:
        #     self._knowledge.clear()
        #     for k, _ in self._knowledge:
        #         v = self._foam_facts[k]
        #         self._knowledge.add(k, u=v.u, v=v.v)

    @property
    def nu(self):
        return self.__nu

    @property
    def rho(self):
        return self.__rho
