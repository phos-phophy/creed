from .inner_models import Base, WOEntities


INNER_MODELS = {
    "base": Base,
    "wo_entities": WOEntities
}


def get_inner_model(model_type: str, pretrained_model_path, **kwargs):
    return INNER_MODELS[model_type](pretrained_model_path, **kwargs)
