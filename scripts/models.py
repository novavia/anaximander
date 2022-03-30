from functools import partial
import time

from typing import ClassVar, Type
import anaximander as nx


@nx.archetype
class MyObject(nx.Object):
    x: str = nx.metadata(max_length=10)

    @x.validator
    def starts_with_x(value: str):
        assert value.startswith("x"), "String must start with 'x'"


@nx.archetype
class DataObject(nx.Object):
    def __init_subclass__(cls) -> None:
        cls._make_parse()

    def __init__(self, data, **kwargs):
        try:
            data = self.parse(data)
        except TypeError:
            print("TypeError")
        else:
            self._data = data

    @classmethod
    def _make_parse(cls):
        def __parse__(cls, data):
            time.sleep(5)
            if data is None:
                raise TypeError
            else:
                return data

        cls.__parse__ = classmethod(__parse__)

    @classmethod
    def parse(cls, data):
        return cls.__parse__(data)

    @property
    def data(self):
        return getattr(self, "_data", None)


@nx.archetype
class Measurement(DataObject):
    quantities: ClassVar[dict[str, str]] = {
        "Celsius": "Temperature",
        "Fahrentheit": "Temperature",
    }
    unit: str = nx.metadata()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        parse = partial(cls.__parse__, cls=cls)
        cls.__parse__ = classmethod(lambda cls, data: parse(data))

    @nx.metaproperty
    def quantity(meta, unit):
        return Measurement.quantities.get(unit, None)

    def __repr__(self):
        if (data := self.data) is not None:
            print(data, self.unit)
            return f"{data:.2f} {self.unit}"
        else:
            return super().__repr__()


CelsiusTemp = Measurement.subtype(unit="Celsius", type_name="CelsiusTemp")
FahrenheitTemp = Measurement.subtype(unit="Fahrenheit", type_name="FahrenheitTemp")
print(CelsiusTemp.quantity)
CelsiusTemp(None)
