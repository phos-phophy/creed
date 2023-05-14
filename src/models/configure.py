from src.abstract import AbstractModel


def get_ssan_adapt():
    from .ssan_adapt import SSANAdapt
    return SSANAdapt


def get_bert_baseline():
    from .bert_baseline import BertBaseline
    return BertBaseline


def get_docunet():
    from .docunet import DocUNet
    return DocUNet


MODELS = {
    "ssan_adapt": get_ssan_adapt,
    "bert_baseline": get_bert_baseline,
    "docunet": get_docunet
}


def get_model(model_type: str, **kwargs) -> AbstractModel:
    return MODELS[model_type]()(**kwargs)
