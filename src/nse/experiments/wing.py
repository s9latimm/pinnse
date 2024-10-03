from src.base.data import arrange, Rectangle, Airfoil
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

        airfoil = Airfoil((1, 1.5), (5, 5))

        geometry = [
            Rectangle((0, 0), (10, 2)),
            airfoil,
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
        for i in arrange(.05, 1.95, .05):
            self._rim_facts.add((0, i), u=flow, v=0)

        # border
        for i in arrange(.05, 9.95, .05):
            self._rim_facts.add((i, 0), u=0, v=0)
            self._rim_facts.add((i, 2), u=0, v=0)

        for i, x in enumerate(airfoil.x[:-1]):
            self._rim_facts.add((x, airfoil.y[i]), u=0, v=0)

        for c in self._finite_elements_mesh:
            if c in airfoil:
                self._rim_facts.add(c, u=0, v=0)

        for c in self._finite_elements_mesh:
            if c not in self._rim_facts:
                self._pde_facts.add(c)

    @property
    def nu(self):
        return self.__nu

    @property
    def rho(self):
        return self.__rho
