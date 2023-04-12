from .configure import get_loader
from .docred import DocREDLoader
from .tacred import TacredLoader

__all__ = [
    "DocREDLoader",
    "TacredLoader",
    "get_loader"
]
