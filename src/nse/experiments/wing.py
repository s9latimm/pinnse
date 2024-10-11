from src.base.mesh import arrange, Mesh, Axis
from src.base.shape import Airfoil, Rectangle, Figure
from src.nse.experiments.experiment import NSEExperiment


class Wing(NSEExperiment):

    def __init__(
        self,
        nu: float,
        rho: float,
        inlet: float,
        _: bool,
    ):
        airfoil = Airfoil((1, 1.25), 5, -10)

        super().__init__(
            'Step',
            Axis('x', 0, 10),
            Axis('y', 0, 2),
            Figure(Rectangle((0, 0), (10, 2))),
            Figure(airfoil),
            nu,
            rho,
            inlet,
        )

        # intake
        for y in arrange(0, 2, .05):
            u = inlet * (1. - (1. - y)**2)
            self._knowledge.emplace((0, y), u=u, v=0)

        # border
        for x in arrange(.05, 9.95, .05):
            self._knowledge.emplace((x, 0), u=0, v=0)
            self._knowledge.emplace((x, 2), u=0, v=0)

        for c in airfoil[::.05]:
            self._knowledge.emplace(c, u=0, v=0)

        iline = airfoil[::.05].interpolate(-.05)

        for c in iline:
            self._knowledge.emplace(c, u=0, v=0)

        mesh = Mesh(self.x.arrange(.1), self.y.arrange(.1))
        for c in mesh:
            if c not in self._knowledge and c in iline:
                self._knowledge.emplace(c, u=0, v=0)

        outline = airfoil[::.05].interpolate(.05)
        bline = outline.interpolate(.05)

        for c in outline:
            if c not in self._learning:
                self._learning.emplace(c)

        for c in bline:
            if c not in self._learning:
                self._learning.emplace(c)

        mesh = Mesh(self.x.arrange(.1, True), self.y.arrange(.1, True))
        for c in mesh:
            if c not in self._knowledge and c not in self._learning and c not in bline:
                self._learning.emplace(c)
