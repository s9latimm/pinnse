import numpy as np
import torch

from src.model import SequentialModel
from src.navier_stokes.geometry import NavierStokesGeometry


class NavierStokesModel(SequentialModel):

    def __init__(self, geometry: NavierStokesGeometry, device, steps):

        layers = [2, 20, 20, 20, 20, 20, 20, 2]

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

        self.__null = torch.zeros(self.__geometry.train.shape[0], 1, dtype=torch.float64, device=self.device)
        self.__u_in = torch.tensor(self.__geometry.intake[:, [2]], dtype=torch.float64, device=self.device)
        self.__v_in = torch.tensor(self.__geometry.intake[:, [3]], dtype=torch.float64, device=self.device)
        self.__u_border = torch.tensor(self.__geometry.border[:, [2]], dtype=torch.float64, device=self.device)
        self.__v_border = torch.tensor(self.__geometry.border[:, [3]], dtype=torch.float64, device=self.device)

    def train(self, callback):

        def closure():
            callback(self.history)
            return self.__loss()

        self._model.train()
        self.__optimizer.step(closure)

    def __loss_pde(self, f, g, m):
        f_loss = self._mse(f, self.__null)
        g_loss = self._mse(g, self.__null)
        m_loss = self._mse(m, self.__null)
        return f_loss, g_loss, m_loss

    def __loss_brdr(self, u, v):
        u_loss = self._mse(u, self.__u_border)
        v_loss = self._mse(v, self.__v_border)
        return u_loss, v_loss

    def __loss_in(self, u, v):
        u_loss = self._mse(u, self.__u_in)
        v_loss = self._mse(v, self.__v_in)
        return u_loss, v_loss

    @property
    def history(self):
        return self.__losses

    def __loss(self):
        self.__optimizer.zero_grad()

        stack = [
            self.__geometry.train,
            self.__geometry.border,
            self.__geometry.intake,
        ]

        stitches = [0]
        for item in stack:
            stitches.append(stitches[-1] + len(item))

        u, v, _, f, g, m = self.predict(np.vstack(stack))

        f_loss, g_loss, m_loss = self.__loss_pde(f[:stitches[1]], g[:stitches[1]], m[:stitches[1]])
        u_brdr_loss, v_brdr_loss = self.__loss_brdr(u[stitches[1]:stitches[2]], v[stitches[1]:stitches[2]])
        u_in_loss, v_in_loss = self.__loss_in(u[stitches[2]:stitches[3]], v[stitches[2]:stitches[3]])

        u_brdr_loss = u_brdr_loss + u_in_loss
        v_brdr_loss = v_brdr_loss + v_in_loss

        loss = f_loss + g_loss + m_loss + u_brdr_loss + v_brdr_loss

        self.__losses = np.vstack([
            self.__losses,
            np.asarray([
                f_loss.detach().cpu().numpy(),
                g_loss.detach().cpu().numpy(),
                m_loss.detach().cpu().numpy(),
                u_brdr_loss.detach().cpu().numpy(),
                v_brdr_loss.detach().cpu().numpy(),
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

        f = u * u_x + v * u_y + p_x - self.__geometry.nu * (u_xx + u_yy)
        g = u * v_x + v * v_y + p_y - self.__geometry.nu * (v_xx + v_yy)
        m = u_x + v_y

        return u, v, p, f, g, m
