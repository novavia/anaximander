from anaximander.meta import (
    TypeParameter,
    Metacharacter,
    archetype,
    nxobject,
    nxtype,
    Object,
    trait,
)


@archetype
class DataObject(Object.Base):
    def __init__(self, data, **kwargs):
        self._data = data


@archetype
class Measurement(DataObject.Base):
    unit = TypeParameter(str)

    def __dir__(self):
        return ["unit", "__init__"]

    def method(self):
        return True


CelsiusTemp = Measurement.subtype(unit="Celsius", type_name="CelsiusTemp")


class FahrenheitTemp(Measurement.Base):
    def to_celsius(self):
        return CelsiusTemp((self._data - 32) / 1.8)


MyObject = Object.subtype()

MyObjectType = type(MyObject)


class C(nxobject, metaclass=nxtype):
    def __init__(self, x):
        pass


M = DataObject.subtype()
