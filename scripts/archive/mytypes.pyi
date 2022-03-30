from typing import Type
from anaximander.meta.metatype import Metatype
from anaximander.meta import (
    nxtype,
    typespec,
    TypeParameter,
    Metacharacter,
    archetype,
    Object,
    trait,
)

class Archetype:
    @property
    def metatype(cls) -> Metatype: ...

class DataObjectType(Archetype):
    @property
    def datatype(cls) -> type: ...

class DataObject_(metaclass=DataObjectType):
    pass

class DataObject(metaclass=DataObjectType):
    def __init__(self, data, **kwargs): ...
    @property
    def data(self): ...

class Measurement(DataObject):
    unit: str
    def method(self) -> bool: ...

class CelsiusTemp(Measurement): ...

class FahrenheitTemp(Measurement):
    def to_celsius(self) -> CelsiusTemp: ...

class KelvinTemp(Measurement): ...

class MyType(type):
    @property
    def mytype(cls) -> type: ...

class MyObject(metaclass=MyType):
    @property
    def myobject(self) -> "MyObject": ...

class Thing:
    @property
    def thing(self) -> bool: ...

class ThingType(type):
    def __call__(mcl, name, bases, namespace) -> Type[Thing]: ...

class Hammer(metaclass=ThingType):
    pass
