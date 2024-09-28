import typing as tp

import numpy as np
import torch
from torch import nn

from src.base.data import Coordinate
from src.base.model import SequentialModel
from src.nse.geometry import NSEGeometry


class NSEModel(SequentialModel):

    def __init__(self, geometry: NSEGeometry, device, steps, supervised=False):

        layers = [2, 20, 20, 20, 20, 20, 2]

        super().__init__(layers, device)

        self.__geometry = geometry

        self.__optimizer = torch.optim.LBFGS(self._model.parameters(),
                                             lr=1,
                                             max_iter=steps,
                                             max_eval=steps,
                                             history_size=50,
                                             tolerance_grad=1e-17,
                                             tolerance_change=1e-17,
                                             line_search_fn="strong_wolfe")
        self.__losses = np.asarray([np.zeros(5)])

        rim = self.__geometry.rim_cloud.detach()
        pde = self.__geometry.pde_cloud.detach()

        self.__null = torch.zeros(len(pde), 1, dtype=torch.float64, device=self.device)
        self.__u = torch.tensor([[i.u] for _, i in rim], dtype=torch.float64, device=self.device)
        self.__v = torch.tensor([[i.v] for _, i in rim], dtype=torch.float64, device=self.device)

        self.__rim = [i for i, _ in rim]
        self.__pde = [i for i, _ in pde]

        if supervised:
            self.__nu = nn.Parameter(data=torch.tensor(1, dtype=torch.float64, device=self.device), requires_grad=True)
            self.__rho = nn.Parameter(data=torch.tensor(1, dtype=torch.float64, device=self.device), requires_grad=True)
        else:
            self.__nu = self.__geometry.nu
            self.__rho = self.__geometry.rho

    def train(self, callback):

        def closure():
            callback(self.history)
            return self.__loss()

        self._model.train()
        self.__optimizer.step(closure)

    def __loss_pde(self):
        *_, f, g, = self.predict(self.__pde, True)
        f_loss = self._mse(f, self.__null)
        g_loss = self._mse(g, self.__null)
        return f_loss, g_loss

    def __loss_rim(self):
        u, v, *_ = self.predict(self.__rim)
        u_loss = self._mse(u, self.__u)
        v_loss = self._mse(v, self.__v)
        return u_loss, v_loss

    @property
    def history(self):
        return self.__losses

    def __loss(self):
        self.__optimizer.zero_grad()

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

    def predict(self, sample: tp.List[Coordinate], nse=False):
        x = torch.tensor([[i.x] for i in sample], dtype=torch.float64, requires_grad=True, device=self.device)
        y = torch.tensor([[i.y] for i in sample], dtype=torch.float64, requires_grad=True, device=self.device)

        res = self._model(torch.hstack([x, y]))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = self.gradient(psi, y)
        v = -self.gradient(psi, x)

        if not nse:
            return u, v, p

        u_x, u_xx = self.derive(u, x)
        u_y, u_yy = self.derive(u, y)

        v_x, v_xx = self.derive(v, x)
        v_y, v_yy = self.derive(v, y)

        p_x = self.gradient(p, x)
        p_y = self.gradient(p, y)

        f = self.__rho * (u * u_x + v * u_y) + p_x - self.__nu * (u_xx + u_yy)
        g = self.__rho * (u * v_x + v * v_y) + p_y - self.__nu * (v_xx + v_yy)

        return u, v, p, f, g
