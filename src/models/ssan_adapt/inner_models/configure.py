from .abstract import AbstractSSANAdaptInnerModel


def get_base():
    from .base import BaseSSANAdaptInnerModel
    return BaseSSANAdaptInnerModel


def get_wo_entities():
    from .wo_types import WOTypesSSANAdaptInnerModel
    return WOTypesSSANAdaptInnerModel


INNER_MODELS = {
    "base": get_base(),
    "wo_entities": get_wo_entities()
}


def get_inner_model(inner_model_type: str, **kwargs) -> AbstractSSANAdaptInnerModel:
    return INNER_MODELS[inner_model_type](**kwargs)
