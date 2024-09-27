import numpy as np
import torch
from torch import nn

from src.model import SequentialModel
from src.navier_stokes.geometry import NavierStokesGeometry


class NavierStokesModel(SequentialModel):

    def __init__(self, geometry: NavierStokesGeometry, device, steps, supervised=False):

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

        self.__null = torch.zeros(self.__geometry.t_stack.shape[0], 1, dtype=torch.float64, device=self.device)
        self.__u = torch.tensor(self.__geometry.b_stack[:, [2]], dtype=torch.float64, device=self.device)
        self.__v = torch.tensor(self.__geometry.b_stack[:, [3]], dtype=torch.float64, device=self.device)

        if supervised:
            self.nu = nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        else:
            self.nu = self.__geometry.nu

    def train(self, callback):

        def closure():
            callback(self.history)
            return self.__loss()

        self._model.train()
        self.__optimizer.step(closure)

    def __loss_pde(self, f, g):
        f_loss = self._mse(f, self.__null)
        g_loss = self._mse(g, self.__null)
        return f_loss, g_loss

    def __loss_brdr(self, u, v):
        u_loss = self._mse(u, self.__u)
        v_loss = self._mse(v, self.__v)
        return u_loss, v_loss

    @property
    def history(self):
        return self.__losses

    def __loss(self):
        self.__optimizer.zero_grad()

        stack = [
            self.__geometry.t_stack,
            self.__geometry.b_stack,
        ]

        split = len(stack[0])

        u, v, _, f, g, = self.predict(np.vstack(stack))

        f_loss, g_loss = self.__loss_pde(f[:split], g[:split])
        u_loss, v_loss = self.__loss_brdr(u[split:], v[split:])

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

    def predict(self, sample):
        x = torch.tensor(sample[:, [0]], dtype=torch.float64, requires_grad=True, device=self.device)
        y = torch.tensor(sample[:, [1]], dtype=torch.float64, requires_grad=True, device=self.device)

        res = self._model(torch.hstack([x, y]))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -1. * torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

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
