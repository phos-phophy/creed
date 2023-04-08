from src.abstract import AbstractModel


def get_ssan_adapt_docred():
    from .ssan_adapt import SSANAdaptDocREDModel
    return SSANAdaptDocREDModel


def get_ssan_adapt_tacred():
    from .ssan_adapt import SSANAdaptTACREDModel
    return SSANAdaptTACREDModel


MODELS = {
    "ssan_adapt_docred": get_ssan_adapt_docred,
    "ssan_adapt_tacred": get_ssan_adapt_tacred,
}


def get_model(model_type: str, **kwargs) -> AbstractModel:
    return MODELS[model_type]()(**kwargs)
