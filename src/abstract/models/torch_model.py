import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import torch

_Model = TypeVar('_Model', bound='TorchModel')


class TorchModel(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        # used to obtain the device of torch module
        self._dummy_param = torch.nn.Parameter(torch.empty(0))

    @property
    def device(self) -> str:
        return self._dummy_param.device

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def to_device(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return tensor.to(device=self.device) if tensor else None

    def save(self, path: Path, *, rewrite: bool = False) -> None:
        previous_device = self.device
        self.cpu()

        if path.exists() and not rewrite:
            raise Exception(f"Model saving path already exists: {path}")

        with path.open('wb') as f:
            pickle.dump(self, f, protocol=4)  # fixed protocol version to avoid issues with serialization on Python 3.6+ versions

        self.to(device=torch.device(previous_device))

    @classmethod
    def load(cls: Type[_Model], path: Path) -> _Model:
        with path.open('rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise Exception(f"Model at {path} is not an instance of {cls}")
        return model

    @classmethod
    def from_config(cls, config: dict) -> _Model:
        return type(cls)(**config)
