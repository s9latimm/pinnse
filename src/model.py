from abc import abstractmethod
from pathlib import Path

import torch
from torch import nn


class SequentialModel:

    def __init__(self, layers, device):
        self.device = torch.device(device)

        modules = [nn.Linear(layers[0], layers[1], bias=True, dtype=torch.float64)]
        for i in range(1, len(layers) - 1):
            modules.append(nn.Tanh())
            modules.append(nn.Linear(layers[i], layers[i + 1], bias=True, dtype=torch.float64))

        self._model = nn.Sequential(*modules)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self._model.apply(init_weights)

        self._model.to(device)

        self._mse = nn.MSELoss()

    def __str__(self):
        return str(self._model)

    @property
    @abstractmethod
    def history(self):
        ...

    @abstractmethod
    def train(self, callback):
        ...

    @abstractmethod
    def predict(self, sample):
        ...

    def eval(self):
        self._model.eval()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)

    def load(self, path: Path):
        self._model.load_state_dict(torch.load(path))
