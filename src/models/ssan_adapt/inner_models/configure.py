from src.abstract import AbstractModel


def get_base():
    from .base import BaseSSANAdaptModel
    return BaseSSANAdaptModel


def get_wo_entities():
    from .wo_types import WOTypesSSANAdaptModel
    return WOTypesSSANAdaptModel


INNER_MODELS = {
    "base": get_base(),
    "wo_entities": get_wo_entities()
}


def get_inner_model(inner_model_type: str, **kwargs) -> AbstractModel:
    return INNER_MODELS[inner_model_type](**kwargs)
