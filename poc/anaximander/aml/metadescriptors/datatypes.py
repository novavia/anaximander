import datetime
import typing
from abc import ABC
from dataclasses import is_dataclass
from enum import Enum
from types import NoneType, UnionType
from typing import ClassVar, Literal

from pydantic import BaseModel


class dataobject(ABC):
    """Abstract base class for all model space data representations.

    These include primitive data types, custom data types that subclass nx.Data,
    structured models -which include subclasses of nx.Model as well as compatible
    model forms including dataclasses and Pydantic models, and collections
    of data and/or models.
    """

    # Set of admissible collection types
    __collections__: ClassVar[set[type]] = {
        list,
        tuple,
        dict,
        set,
    }

    @classmethod
    def validate_hint(cls, hint):
        """Validates typing hint compatibility at runtime.

        Note that hint is expected to be the pulled from the return value of typing.get_type_hints,
        which differs slightly from the built-in __annotations__ dictionary, in particular with
        regards to forward references.
        """
        if isinstance(hint, type):
            if issubclass(hint, Enum):
                values = [m._value_ for m in hint.__members__.values()]
                vtypes = {type(v) for v in values}
                return all(cls.validate_hint(t) for t in vtypes)
            return issubclass(hint, cls)
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)
        if origin in cls.__collections__:
            return all(cls.validate_hint(arg) for arg in args)
        elif origin in (UnionType, typing.Union):
            args = set(args)
            args.discard(NoneType)
            return all(cls.validate_hint(arg) for arg in args)
        elif origin is Literal:
            if len(args) == 1:
                return cls.validate_hint(type(args[0]))
            args = set(args)
            args.discard(None)
            return all(cls.validate_hint(type(arg)) for arg in args)
        return False


class data(dataobject):
    """Abstract base class for primitive/scalar data types."""

    __primitives__: ClassVar[set[type]] = {
        bool,
        int,
        float,
        str,
        bytes,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
    }


for cls in data.__primitives__:
    data.register(cls)


class model(dataobject):
    """Abastract base class for structured data types."""

    @classmethod
    def __subclasshook__(cls, subclass):
        if is_dataclass(subclass):
            return True
        elif issubclass(subclass, BaseModel):
            return True
        return NotImplemented
