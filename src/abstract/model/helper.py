from functools import wraps
from typing import Callable

import torch
from torch import amp


def cuda_autocast(forward: Callable):
    @wraps(forward)
    def amp_wrapper(self: torch.nn.Module, *args, **kwargs):
        if self.device.type == 'cuda':
            with amp.autocast(device_type='cuda'):
                return forward(self, *args, **kwargs)
        return forward(self, *args, **kwargs)

    return amp_wrapper
