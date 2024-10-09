import numpy as np
import torch
from torch import nn

from src.base.mesh import Coordinate
from src.base.model import SequentialModel
from src.nse.experiments.experiment import NSEExperiment


class NSEModel(SequentialModel):

    def __init__(self, experiment: NSEExperiment, device: str, steps: int, layers: list[int]) -> None:

        super().__init__([2] + layers + [2], device)

        self.__experiment = experiment

        rim = self.__experiment.knowledge.detach()
        pde = self.__experiment.learning.detach()

        self.__null = torch.zeros(len(pde), 1, dtype=torch.float64, device=self.device)
        self.__u = torch.tensor([[i.u] for _, i in rim], dtype=torch.float64, device=self.device)
        self.__v = torch.tensor([[i.v] for _, i in rim], dtype=torch.float64, device=self.device)

        self.__rim = [i for i, _ in rim]
        self.__pde = [i for i, _ in pde]

        self.__nu = torch.tensor(self.__experiment.nu, dtype=torch.float64, device=self.device)
        self.__rho = torch.tensor(self.__experiment.rho, dtype=torch.float64, device=self.device)

        if experiment.supervised:
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
        self.__losses = np.asarray([np.zeros(5)])

    def __len__(self) -> int:
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @property
    def nu(self) -> float:
        return self.__nu.detach().cpu()

    @property
    def rho(self) -> float:
        return self.__rho.detach().cpu()

    def train(self, callback) -> None:

        def closure():
            callback(self.history)
            return self.__loss()

        self._model.train()
        self.__optimizer.step(closure)

    def __loss_pde(self) -> tuple[torch.Tensor, torch.Tensor]:
        *_, f, g = self.predict(self.__pde, True)
        f_loss = self._mse(f, self.__null)
        g_loss = self._mse(g, self.__null)
        return f_loss, g_loss

    def __loss_rim(self) -> tuple[torch.Tensor, torch.Tensor]:
        u, v, *_ = self.predict(self.__rim)
        u_loss = self._mse(u, self.__u)
        v_loss = self._mse(v, self.__v)
        return u_loss, v_loss

    @property
    def history(self):
        return self.__losses

    def __loss(self):
        self.__optimizer.zero_grad()

        # with torch.no_grad():
        #     weights = parameters_to_vector(self._model.parameters())
        #     weights.add_(1e-14 * torch.randn(len(weights), dtype=torch.float64))
        #     vector_to_parameters(weights, self._model.parameters())

        f_loss, g_loss = self.__loss_pde()
        u_loss, v_loss = self.__loss_rim()

        loss = f_loss + g_loss + u_loss + v_loss

        self.__losses = np.vstack([
            self.__losses,
            np.array([
                f_loss.detach().cpu().numpy(),
                g_loss.detach().cpu().numpy(),
                u_loss.detach().cpu().numpy(),
                v_loss.detach().cpu().numpy(),
                loss.detach().cpu().numpy(),
            ]),
        ])

        loss.backward()
        return loss

    def predict(
        self,
        sample: list[Coordinate],
        nse=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                                 torch.Tensor]:
        x = torch.tensor([[i.x] for i in sample], dtype=torch.float64, requires_grad=True, device=self.device)
        y = torch.tensor([[i.y] for i in sample], dtype=torch.float64, requires_grad=True, device=self.device)

        res = self._model(torch.hstack([x, y]))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = self.nabla(psi, y)
        v = -self.nabla(psi, x)

        # u, v, p = res[:, 0:1], res[:, 1:2], res[:, 2:3]

        if not nse:
            return u, v, p

        u_x, u_xx = self.laplace(u, x)
        u_y, u_yy = self.laplace(u, y)

        v_x, v_xx = self.laplace(v, x)
        v_y, v_yy = self.laplace(v, y)

        p_x = self.nabla(p, x)
        p_y = self.nabla(p, y)

        f = self.__rho * (u * u_x + v * u_y - self.__nu * (u_xx + u_yy)) + p_x
        g = self.__rho * (u * v_x + v * v_y - self.__nu * (v_xx + v_yy)) + p_y

        return u, v, p, f, g
