import typing as tp

import numpy as np
import torch
from torch import nn

from src.base.data import Coordinate
from src.base.model import SequentialModel
from src.nse.geometry import NSEGeometry


class NSEModel(SequentialModel):

    def __init__(self, geometry: NSEGeometry, device, steps, rim=False):

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

        pde = self.__geometry.pde_cloud.detach()
        rim = self.__geometry.rim_cloud.detach()

        self.__null = torch.zeros(len(pde), 1, dtype=torch.float64, device=self.device)
        self.__u = torch.tensor([[i.u] for _, i in rim], dtype=torch.float64, device=self.device)
        self.__v = torch.tensor([[i.v] for _, i in rim], dtype=torch.float64, device=self.device)

        self.__rim = [i for i, _ in rim]
        self.__pde = [i for i, _ in pde]

        if rim:
            self.nu = nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        else:
            self.nu = self.__geometry.nu

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
            np.asarray([
                f_loss.detach().cpu().numpy(),
                g_loss.detach().cpu().numpy(),
                u_loss.detach().cpu().numpy(),
                v_loss.detach().cpu().numpy(),
                loss.detach().cpu().numpy(),
            ]),
        ])

        loss.backward()
        return loss

    def predict(self, sample: tp.List[Coordinate], pde=False):
        x = torch.tensor([[i.x] for i in sample], dtype=torch.float64, requires_grad=True, device=self.device)
        y = torch.tensor([[i.y] for i in sample], dtype=torch.float64, requires_grad=True, device=self.device)

        res = self._model(torch.hstack([x, y]))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -1. * torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        if not pde:
            return u, v, p

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        g = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        return u, v, p, f, g
