from .meta import DataABC, Prototype


class DataType(Prototype):
    """Metaclass for Data and its subclasses."""

    pass


class Data(DataABC, metaclass=DataType):
    """Base class for primitive data type declarations"""

    pass
