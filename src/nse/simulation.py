import typing as tp

import torch
from torch import nn

from src.base.mesh import Mesh
from src.base.model import SequentialModel, laplace, nabla
from src.nse.experiments.experiment import Experiment
from src.nse.record import Record


class Simulation(SequentialModel[Record]):

    def __init__(self, experiment: Experiment, device: str, steps: int, layers: list[int]) -> None:

        self.__experiment = experiment

        super().__init__([2] + layers + [2], device)

        knowledge = self.__experiment.knowledge.detach()
        learning = self.__experiment.learning.detach()
        inlet = self.__experiment.inlet.detach()
        outlet = self.__experiment.outlet.detach()

        self.__u = torch.tensor([[i.u] for _, i in inlet + knowledge], dtype=torch.float64, device=self._device)
        self.__v = torch.tensor([[i.v] for _, i in inlet + knowledge], dtype=torch.float64, device=self._device)

        self.__null = torch.zeros(len(learning), 1, dtype=torch.float64, device=self._device)

        self.__out = len(outlet)
        self.__u_out = torch.zeros(len(outlet), 1, dtype=torch.float64, device=self._device)

        self.__knowledge = (
            torch.tensor([[k.x] for k, _ in outlet + inlet + knowledge],
                         dtype=torch.float64,
                         requires_grad=True,
                         device=self._device),
            torch.tensor([[k.y] for k, _ in outlet + inlet + knowledge],
                         dtype=torch.float64,
                         requires_grad=True,
                         device=self._device),
        )
        self.__learning = (
            torch.tensor([[k.x] for k, _ in learning], dtype=torch.float64, requires_grad=True, device=self._device),
            torch.tensor([[k.y] for k, _ in learning], dtype=torch.float64, requires_grad=True, device=self._device),
        )

        self.__nu = torch.tensor(self.__experiment.nu, dtype=torch.float64, device=self._device)
        self.__rho = torch.tensor(self.__experiment.rho, dtype=torch.float64, device=self._device)

        if self.__experiment.supervised:
            self.__nu = nn.Parameter(self.__nu, requires_grad=True)
            self._model.register_parameter('my', self.__nu)

            # self.__rho = nn.Parameter(self.__rho, requires_grad=True)
            # self._model.register_parameter('rho', self.__rho)

        self.__optimizer = torch.optim.LBFGS(self._model.parameters(),
                                             lr=1,
                                             max_iter=steps,
                                             max_eval=steps,
                                             history_size=50,
                                             tolerance_grad=1e-17,
                                             tolerance_change=5e-12,
                                             line_search_fn="strong_wolfe")

        self._losses.append([0, 0, 0, 0, 0])

    def __len__(self) -> int:
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @property
    def nu(self) -> float:
        return self.__nu.detach().cpu()

    @property
    def rho(self) -> float:
        return self.__rho.detach().cpu()

    def train(self, callback: tp.Callable[[tp.Any], None]) -> None:

        def closure():
            callback(self.history)
            return self.__loss()

        self._model.train()
        self.__optimizer.step(closure)

    def __loss(self):
        self.__optimizer.zero_grad()

        u, v, *_ = self.__forward(self.__knowledge)

        u_loss = self._mse(u[self.__out:], self.__u)
        v_loss = self._mse(v[self.__out:], self.__v)

        # prohibits the model from hallucinating an incoming flow from right
        if self.__out > 0:
            u_loss += self._mse(torch.clamp(u[:self.__out], max=0), self.__u_out)

        *_, f, g = self.__forward(self.__learning, True)

        f_loss = self._mse(f, self.__null)
        g_loss = self._mse(g, self.__null)

        loss = f_loss + g_loss + u_loss + v_loss

        self._losses.append(self._detach(f_loss, g_loss, u_loss, v_loss, loss))

        loss.backward()
        return loss

    def __forward(
        self,
        t: tuple[torch.Tensor, torch.Tensor],
        nse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        x, y = t

        res = self._model.forward(torch.hstack([x, y]))

        psi, p = res[:, 0:1], res[:, 1:2]

        u = nabla(psi, y)
        v = -nabla(psi, x)

        if not nse:
            return u, v, p, psi

        u_x, u_xx = laplace(u, x)
        u_y, u_yy = laplace(u, y)

        v_x, v_xx = laplace(v, x)
        v_y, v_yy = laplace(v, y)

        p_x = nabla(p, x)
        p_y = nabla(p, y)

        f = self.__rho * (u * u_x + v * u_y - self.__nu * (u_xx + u_yy)) + p_x
        g = self.__rho * (u * v_x + v * v_y - self.__nu * (v_xx + v_yy)) + p_y

        return f, g

    def predict(self, mesh: Mesh) -> Mesh[Record]:
        coordinates = [k for k, _ in mesh.detach()]

        u, v, p, _ = self._detach(*self.__forward((
            torch.tensor([[i.x] for i in coordinates], dtype=torch.float64, requires_grad=True, device=self._device),
            torch.tensor([[i.y] for i in coordinates], dtype=torch.float64, requires_grad=True, device=self._device),
        )))

        prediction = Mesh(Record)
        for i, c in enumerate(coordinates):
            prediction.insert(c, Record(u[i][0], v[i][0], p[i][0]))

        return prediction
