from ..metadescriptors import datatypes
from .base import Prototype


class DataType(Prototype):
    """Metaclass for Data and its subclasses."""

    pass


class Data(datatypes.Data, metaclass=DataType):
    pass
