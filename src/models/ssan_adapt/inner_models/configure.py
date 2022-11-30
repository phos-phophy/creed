from .model import AbstractInnerModel


def get_base():
    from .base import Base
    return Base


def get_wo_entities():
    from .wo_entities import WOEntities
    return WOEntities


INNER_MODELS = {
    "base": get_base(),
    "wo_entities": get_wo_entities()
}


def get_inner_model(model_type: str, pretrained_model_path, **kwargs) -> AbstractInnerModel:
    return INNER_MODELS[model_type](pretrained_model_path, **kwargs)
