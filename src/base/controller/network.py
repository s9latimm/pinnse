import typing as tp
from abc import abstractmethod
from pathlib import Path

import torch
from torch import nn

from src.base.model.mesh import Mesh


def nabla(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, retain_graph=True)[0]


def laplace(f: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    f_x = nabla(f, x)
    f_xx = nabla(f_x, x)
    return f_x, f_xx


T = tp.TypeVar('T')


class SequentialModel(tp.Generic[T]):

    def __init__(self, layers: tp.Sequence[int], device: str) -> None:
        super().__init__()

        assert torch.cpu.is_available()
        if device == 'cuda':
            assert torch.cuda.is_available()

        self._device = torch.device(device)

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

        self._losses = []

    @staticmethod
    def _detach(*tensor: torch.Tensor) -> tuple[list[list[float]], ...]:
        return tuple(i.detach().cpu().tolist() for i in tensor)

    def __str__(self) -> str:
        return str(self._model)

    @property
    def losses(self) -> list[list[tp.Any]]:
        return self._losses

    @abstractmethod
    def train(self, callback: tp.Callable[[tp.Any], tp.Any]) -> None:
        ...

    @abstractmethod
    def predict(self, mesh: Mesh) -> Mesh[T]:
        ...

    def eval(self) -> None:
        self._model.eval()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for i, loss in enumerate(self._losses):
                s = ",".join(f'{j:.16f}' for j in loss)
                f.write(f'{i:d},{s}\n')
