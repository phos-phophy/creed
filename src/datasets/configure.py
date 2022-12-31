from src.abstract import AbstractConverter


def get_docred_converter():
    from .docred_converter import DocREDConverter
    return DocREDConverter


CONVERTERS = {
    "docred": get_docred_converter(),
}


def get_converter(converter_type: str, **kwargs) -> AbstractConverter:
    return CONVERTERS[converter_type](**kwargs)
