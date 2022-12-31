from src.abstract import AbstractModel


def get_ssan_adapt():
    from .ssan_adapt import SSANAdaptModel
    return SSANAdaptModel


MODELS = {
    "ssan_adapt": get_ssan_adapt(),
}


def get_model(model_type: str, **kwargs) -> AbstractModel:
    return MODELS[model_type](**kwargs)
