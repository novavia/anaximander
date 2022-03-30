from anaximander.meta.metatype import Metatype
from anaximander.meta import (
    nxtype,
    typespec,
    TypeParameter,
    Metacharacter,
    archetype,
    Archetype,
    Object,
    trait,
)


@archetype
class DataObject(Object):
    def __init__(self, data, **kwargs):
        self._data = data


class DataObject_:
    pass


@archetype
class Measurement(DataObject):
    unit = TypeParameter(str)

    def method(self):
        return True


CelsiusTemp = Measurement.subtype(unit="Celsius", type_name="CelsiusTemp")


_FahrenheitTemp = Measurement.subtype(unit="Fahrenheit", type_name="FahrenheitTemp")


class FahrenheitTemp(_FahrenheitTemp):
    def to_celsius(self):
        return CelsiusTemp((self._data - 32) / 1.8)


class KelvinTemp(Measurement):
    unit = "Kelvin"


class MyType(type):
    pass


class MyObject(metaclass=MyType):
    pass


class Thing:
    @property
    def thing(self):
        return True


class ThingType(type):
    pass


class Hammer(metaclass=ThingType):
    pass
