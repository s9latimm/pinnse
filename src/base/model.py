import typing as tp
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SequentialModel:

    @staticmethod
    def nabla(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    @staticmethod
    def laplace(f: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_x = SequentialModel.nabla(f, x)
        f_xx = SequentialModel.nabla(f_x, x)
        return f_x, f_xx

    def __init__(self, layers: tp.Sequence[int], device: str) -> None:
        if device == 'cpu':
            assert torch.cpu.is_available()
        if device == 'cuda':
            assert torch.cuda.is_available()

        self.device = torch.device(device)

        self._model = nn.Sequential()
        self._model.append(nn.Linear(layers[0], layers[1], bias=True, dtype=torch.float64))
        for i in range(1, len(layers) - 1):
            self._model.append(nn.Tanh())
            self._model.append(nn.Linear(layers[i], layers[i + 1], bias=True, dtype=torch.float64))

        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #
        # self._model.apply(init_weights)

        self._model.to(device)

        self._mse = nn.MSELoss()

    def __str__(self) -> str:
        return str(self._model)

    @property
    @abstractmethod
    def history(self) -> np.ndarray:
        ...

    @abstractmethod
    def train(self, callback) -> None:
        ...

    @abstractmethod
    def predict(self, sample) -> tuple:
        ...

    def eval(self) -> None:
        self._model.eval()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)

    def load(self, path: Path) -> None:
        self._model.load_state_dict(torch.load(path))
