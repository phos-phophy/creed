from src.abstract import AbstractLoader


def get_docred_loader():
    from .docred import DocREDLoader
    return DocREDLoader


LOADERS = {
    "docred": get_docred_loader(),
}


def get_loader(loader_type: str, **kwargs) -> AbstractLoader:
    return LOADERS[loader_type](**kwargs)
