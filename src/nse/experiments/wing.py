from src.base.geometry import arrange, Mesh
from src.base.shapes import Airfoil
from src.nse.experiments.experiment import Axis, NSEExperiment


class Wing(NSEExperiment):

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

        airfoil = Airfoil((1, 1.25), (5, 5), -10)

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
            [airfoil],
        )

        # intake
        for y in arrange(.05, 1.95, .05):
            self._knowledge.add((0, y), u=flow, v=0)

        # border
        for x in arrange(.05, 9.95, .05):
            self._knowledge.add((x, 0), u=0, v=0)
            self._knowledge.add((x, 2), u=0, v=0)

        for c in airfoil[::.05]:
            self._knowledge.add(c, u=0, v=0)

        iline = airfoil[::.05].interpolate(-.05)

        for c in iline:
            self._knowledge.add(c, u=0, v=0)

        mesh = Mesh(self.x.arrange(.1, True), self.y.arrange(.1, True))
        for c in mesh:
            if c not in self._knowledge and c in iline:
                self._knowledge.add(c, u=0, v=0)

        outline = airfoil[::.05].interpolate(.05)
        bline = airfoil[::.05].interpolate(.1)

        for c in outline:
            if c not in self._learning:
                self._learning.add(c)

        mesh = Mesh(self.x.arrange(.1, True), self.y.arrange(.1, True))
        for c in mesh:
            if c not in self._knowledge and c not in self._learning and c not in bline:
                self._learning.add(c)

    @property
    def nu(self):
        return self.__nu

    @property
    def rho(self):
        return self.__rho
