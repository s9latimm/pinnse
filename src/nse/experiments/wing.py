from src.base.mesh import arrange, Mesh
from src.base.shape import Airfoil, Rectangle, Figure
from src.nse.experiments.experiment import Axis, NSEExperiment


class Wing(NSEExperiment):

    def __init__(
        self,
        nu: float,
        rho: float,
        inlet: float,
        foam: bool,
        supervised: bool,
    ):
        airfoil = Airfoil((1, 1.25), 5, -10)

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
            Figure(airfoil),
        )

        # intake
        for y in arrange(0, 2, .05):
            u = inlet * (1. - (1. - y)**2)
            self._knowledge.add((0, y), u=u, v=0)

        # border
        for x in arrange(.05, 9.95, .05):
            self._knowledge.add((x, 0), u=0, v=0)
            self._knowledge.add((x, 2), u=0, v=0)

        for c in airfoil[::.05]:
            self._knowledge.add(c, u=0, v=0)

        iline = airfoil[::.05].interpolate(-.05)

        for c in iline:
            self._knowledge.add(c, u=0, v=0)

        mesh = Mesh(self.x.arrange(.1), self.y.arrange(.1))
        for c in mesh:
            if c not in self._knowledge and c in iline:
                self._knowledge.add(c, u=0, v=0)

        outline = airfoil[::.05].interpolate(.05)
        bline = outline.interpolate(.05)

        for c in outline:
            if c not in self._learning:
                self._learning.add(c)

        for c in bline:
            if c not in self._learning:
                self._learning.add(c)

        mesh = Mesh(self.x.arrange(.1, True), self.y.arrange(.1, True))
        for c in mesh:
            if c not in self._knowledge and c not in self._learning and c not in bline:
                self._learning.add(c)
