from src.abstract import AbstractModel


def get_base():
    from .base import BaseSSANAdaptModel
    return BaseSSANAdaptModel


def get_wo_entities():
    from .wo_entities import WOEntities
    return WOEntities


INNER_MODELS = {
    "base": get_base(),
    "wo_entities": get_wo_entities()
}


def get_inner_model(model_type: str, **kwargs) -> AbstractModel:
    return INNER_MODELS[model_type](**kwargs)
