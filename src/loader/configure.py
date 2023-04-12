from src.abstract import AbstractLoader


def get_docred_loader():
    from .docred import DocREDLoader
    return DocREDLoader


def get_tacred_loader():
    from .tacred import TacredLoader
    return TacredLoader


LOADERS = {
    "docred": get_docred_loader,
    "tacred": get_tacred_loader
}


def get_loader(loader_type: str, **kwargs) -> AbstractLoader:
    return LOADERS[loader_type]()(**kwargs)
