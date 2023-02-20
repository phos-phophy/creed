from .abstract import AbstractSSANAdaptInnerModel


def get_base():
    from .base import BaseSSANAdaptInnerModel
    return BaseSSANAdaptInnerModel


def get_wo_entities():
    from .wo_types import WOTypesSSANAdaptInnerModel
    return WOTypesSSANAdaptInnerModel


def get_ie_entities():
    # from .ie_types import IETypesSSANAdaptInnerModel
    # return IETypesSSANAdaptInnerModel
    raise NotImplementedError


INNER_MODELS = {
    "base": get_base(),
    "wo": get_wo_entities(),
    "ie": get_ie_entities()
}


def get_inner_model(inner_model_type: str, **kwargs) -> AbstractSSANAdaptInnerModel:
    return INNER_MODELS[inner_model_type](**kwargs)
