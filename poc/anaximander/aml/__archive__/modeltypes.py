from abc import ABCMeta
from dataclasses import is_dataclass

from pydantic import BaseModel

from .prototypes import Prototype


class modelmeta(type):
    """Metaclass for modeltype to define __instancecheck__."""

    def __instancecheck__(mcl, subclass):
        supercheck = super().__instancecheck__(subclass)
        if supercheck is True:
            return True
        else:
            return issubclass(subclass, model)


class modeltype(ABCMeta, metaclass=modelmeta):
    """Abstract base metaclass for model types."""
    pass


class model(metaclass=modeltype):
    """Abstract base class for model types."""

    @classmethod
    def __subclasshook__(cls, subclass):
        if is_dataclass(subclass):
            return True
        elif issubclass(subclass, BaseModel):
            return True
        return NotImplemented


class ModelType(Prototype, modeltype):
    """Metaclass for model classes."""
    pass


class Model(model, metaclass=ModelType):
    pass
