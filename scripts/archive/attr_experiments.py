from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, ClassVar, Optional, Type


import attr
from pydantic import BaseModel
from dataclasses import dataclass, field, MISSING, make_dataclass
from pyfields import field as pyfield, make_init


class Wall:
    height = pyfield(doc="Height of the wall in mm.")
    color = pyfield(default="white", doc="Color of the wall.")
    __init__ = make_init()


w = Wall(1, color="blue")
assert vars(w) == {"color": "blue", "height": 1}


@attr.s(frozen=True)
class C:
    y: ClassVar[int] = 0
    x: int = attr.ib()


@attr.s
class X:
    x: int = attr.ib()


class D:
    def __init__(self, x):
        self.x = x


class E(BaseModel):
    c: ClassVar[int] = 0
    x: float = ...
    y: Optional[float] = None
    z: float = None


@dataclass
class F:
    x: int
    y: ClassVar[int] = 0


class StructuralDescriptor(ABC):
    @abstractmethod
    def __fill__(self):
        return None

    @abstractmethod
    def __cast__(self, value):
        return value

    @abstractmethod
    def __validate__(self, value):
        return True

    def fill(self):
        return self.__fill__()

    def cast(self, value):
        return self.__cast__(value)

    def validate(self, value):
        try:
            assert self.__validate__(value)
        except AssertionError:
            raise ValueError()


@dataclass
class DataDescriptor(StructuralDescriptor):
    name: str
    type: Type
    default: Any = MISSING
    converter: Optional[Callable] = None
    validator: Optional[Callable] = None
    metadata: Optional[dict] = None

    def __fill__(self):
        if default := self.default is MISSING:
            raise ValueError
        return default

    def __cast__(self, value):
        if converter := self.converter is None:
            return value
        return converter(value)

    def __validate__(self, value):
        if validator := self.validator is None:
            return True
        return validator(value)

    def to_datafield(self):
        if callable(self.default):
            default = MISSING
            default_factory = self.default
        else:
            default = self.default
            default_factory = MISSING
        return field(
            default=default,
            default_factory=default_factory,
            repr=False,
            init=True,
            compare=True,
            metadata=None,
        )


@dataclass
class Field(DataDescriptor):
    pass


class DataType(ABCMeta):
    def __new__(mcl, name, bases, namespace, **kwargs):
        # This avoids recursion because make_dataclass calls __new__ again
        try:
            assert namespace.get("__datatyped__")
        except AssertionError:
            cls = super().__new__(mcl, name, bases, namespace)
            if not bases:
                return cls
            cls.__descriptors__ = cls.__descriptors__.copy()
            datafields = []
            for k, v in namespace.items():
                if isinstance(v, DataDescriptor):
                    v.name = k
                    cls.__descriptors__[k] = v
                    datafields.append((k, v.type, v.to_datafield()))
            cls = make_dataclass(
                cls.__name__,
                datafields,
                bases=(cls,),
                namespace={"__datatyped__": True},
            )
            return cls
        else:
            return super().__new__(mcl, name, bases, namespace, **kwargs)


class DataObject(metaclass=DataType):
    __descriptors__ = {}


class Model(DataObject):
    x = Field("x", type=int, default=None)


m = Model(0)

c = C("abc")
