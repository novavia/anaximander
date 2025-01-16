from .base import Prototype
from ..metadescriptors import data


class DataType(Prototype):
    """Metaclass for Data and its subclasses."""
    pass


class Data(data, metaclass=DataType):
    pass
