from abc import ABCMeta
import datetime

from .metadescriptors import annotated_metadescriptor, metadescriptor
from .prototypes import Prototype


class datameta(type):
    """Metaclass for datatype to define __instancecheck__.
    
    datatype is a broad, symbolic metatype for all model-space representations,
    including primitive data types, models and collections of data types.
    """

    def __instancecheck__(mcl, type_):
        supercheck = super().__instancecheck__(type_)
        if supercheck is True:
            return True
        match type_:
            case type():
            return issubclass(subclass, data)


class datatype(ABCMeta, metaclass=datameta):
    """Abstract base metaclass for data types."""
    __primitives__  = [bool, 
                       int, 
                       float, 
                       str,
                       bytes,
                       datetime.date,
                       datetime.datetime,
                       datetime.time,
                       datetime.timedelta,
                       ]


class data(metaclass=datatype):
    """Abstract base class for primitive data types."""

    @classmethod
    def __subclasshook__(cls, subclass):
        return NotImplemented

for dt in datatype.__primitives__:
    data.register(dt)


class DataType(Prototype, datatype):
    """Metaclass for Data and its subclasses."""
    pass


class Data(data, metaclass=DataType):
    pass


class datatype_metadescriptor(annotated_metadescriptor):
    """Base class for metadescriptors that must be subtypes of datatype."""

    def validate_annotation(self, annotation):
        if isinstance(annotation, type):
            return isinstance(annotation, datatype)