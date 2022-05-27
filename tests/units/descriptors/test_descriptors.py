from typing import ClassVar, Optional
import pytest

from pydantic import ValidationError
from pydantic.fields import FieldInfo

from anaximander.descriptors.base import Descriptor, DescriptorRegistry, SettingRegistry
from anaximander.descriptors.fields import DataField, MetadataField, OptionField
from anaximander.descriptors.schema import Schema, DataSchema
from anaximander.descriptors import data
from anaximander.utilities.functions import method_arguments


@pytest.fixture(scope="module")
def SimpleObject():
    class SimpleObject:
        __descriptors__ = DescriptorRegistry()
        x: int = DataField()
        print_hello: bool = OptionField(False)

        def __init__(self, x, print_hello=False):
            settings = method_arguments()
            self._settings = SettingRegistry(self.__descriptors__)
            self._settings.collect(settings)
            self.x = x
            self.print_hello = print_hello
            if print_hello:
                print("hello")

        @x.validator
        def anything_goes(cls, value: int):
            return True

    SimpleObject.__descriptors__.collect(SimpleObject)

    return SimpleObject


@pytest.fixture(scope="module")
def DerivedObject(SimpleObject):
    class DerivedObject(SimpleObject):
        x: float = MetadataField()

    Descriptor.set_up_registry(DerivedObject)
    DerivedObject.__descriptors__.collect(DerivedObject)

    return DerivedObject


@pytest.fixture(scope="module")
def OtherDerivedObject(SimpleObject):
    class OtherDerivedObject(SimpleObject):
        y: float = MetadataField()

    Descriptor.set_up_registry(OtherDerivedObject)
    OtherDerivedObject.__descriptors__.collect(OtherDerivedObject)

    return OtherDerivedObject


@pytest.fixture(scope="module")
def ComplexObject():
    class ComplexObject:
        ten: ClassVar[int] = 10
        __descriptors__ = DescriptorRegistry()
        parsed: int = DataField()
        twice_validated: int = DataField()
        optional: Optional[str] = DataField()
        collection: list[int] = DataField()

        @parsed.parser
        def to_integer(cls, value):
            return abs(hash(value))

        @twice_validated.validator
        def greater_than_zero(cls, value):
            "Value must be greater than 0"
            return value >= 0

        @twice_validated.validator
        def less_than_ten(cls, value):
            "Value must be less than ten"
            return value <= cls.ten

        @data.validator
        def parsed_more_than_twice_validated(cls, values):
            parsed = values.get("parsed")
            twice_validated = values.get("twice_validated")
            return parsed > twice_validated

    ComplexObject.__descriptors__.collect(ComplexObject)

    return ComplexObject


def test_attribute_validation():
    with pytest.raises(ValidationError):
        DataField(pydantic_validators="Eat this!")


def test_field_interface(SimpleObject, ComplexObject):
    x = SimpleObject.x
    assert isinstance(x, DataField)
    assert x.name == "x"
    assert x.type_ is int
    assert SimpleObject.anything_goes(0)
    print_hello = SimpleObject.print_hello
    assert print_hello.default == False
    assert ComplexObject.optional.nullable
    assert ComplexObject.optional.dispensable


X_STRING = """!datafield
name: x
datatype: int
validators:
  custom: '!anything_goes'
"""

PRINT_HELLO_STRING = """!optionfield
name: print_hello
datatype: bool
default: false
"""


def test_field_printing(SimpleObject):
    x = SimpleObject.x
    assert str(x) == X_STRING
    print_hello = SimpleObject.print_hello
    assert str(print_hello) == PRINT_HELLO_STRING


def test_registration(SimpleObject):
    assert DataField.handles == {"data", "datafield", "datafields"}
    registry = SimpleObject.__descriptors__
    assert isinstance(registry, DescriptorRegistry)
    assert registry["data"]["x"] == SimpleObject.x


def test_handles(SimpleObject):
    registry = SimpleObject.__descriptors__
    assert isinstance(registry["option"]["print_hello"], OptionField)
    assert registry["option"] == registry["options"]


def test_registration_overwrite(DerivedObject):
    registry = DerivedObject.__descriptors__
    assert not registry["data"]
    assert registry["metadata"]["x"] == DerivedObject.x


def test_registry_fetch(SimpleObject, DerivedObject):
    registry: DescriptorRegistry = DerivedObject.__descriptors__
    assert registry.fetch() == {"x": DerivedObject.x}
    assert set(registry.fetch(recursive=True)) == {"x", "print_hello"}
    assert not registry.fetch("data")
    assert not registry.fetch("data", recursive=True)
    assert not registry.fetch("options")
    assert list(registry.fetch("options", recursive=True).values()) == [
        SimpleObject.print_hello
    ]
    assert list(registry.parent.fetch("data").values()) == [SimpleObject.x]
    assert list(registry.parent.fetch("data", recursive=True).values()) == [
        SimpleObject.x
    ]
    assert list(registry.fetch("data", "metadata", recursive=True).values()) == [
        DerivedObject.x
    ]
    registry = SimpleObject.__descriptors__
    assert list(
        registry.fetch(recursive=True, filter=lambda d: bool(d.validators))
    ) == ["x"]
    assert list(registry.fetch(recursive=True, type_=bool)) == ["print_hello"]


def test_settings_registration(SimpleObject, DerivedObject):
    simple = SimpleObject(x=0)
    assert simple._settings["data"] == {"x": 0}
    assert simple._settings.fetch() == {"x": 0, "print_hello": False}
    derived = DerivedObject(x=0)
    assert derived._settings["data"] == {}
    assert derived._settings["metadata"] == {"x": 0}
    assert simple._settings.fetch() == {"x": 0, "print_hello": False}


def test_pydantic_field_factory(SimpleObject):
    print_hello: OptionField = SimpleObject.print_hello
    pydantic_print_hello = print_hello.make_pydantic_field_info()
    assert isinstance(pydantic_print_hello, FieldInfo)


def test_schema(SimpleObject, ComplexObject):
    fields = SimpleObject.__descriptors__.fetch()
    schema = Schema(fields=fields.values())
    assert set(schema.fields) == {"x", "print_hello"}
    fields = ComplexObject.__descriptors__.fetch("datafields")
    validators = [
        d.method for d in ComplexObject.__descriptors__.fetch("datavalidators").values()
    ]
    schema = DataSchema(fields=fields.values(), validators=validators)
    assert schema.dispensable == {"optional"}


COMPLEX_OBJECT_SCHEMA_STRING = """!dataschema
name: DataSchema
fields:
  parsed:
    datatype: int
    parser: '!to_integer'
  twice_validated:
    datatype: int
    validators:
      custom: '!greater_than_zero'
      custom-2: '!less_than_ten'
  optional:
    datatype: str
    nullable: true
  collection:
    datatype: list
validators:
  custom: '!parsed_more_than_twice_validated'
"""


def test_schema_printing(ComplexObject):
    fields = ComplexObject.__descriptors__.fetch("datafields")
    validators = [
        d.method for d in ComplexObject.__descriptors__.fetch("datavalidators").values()
    ]
    schema = DataSchema(fields=fields.values(), validators=validators)
    assert str(schema) == COMPLEX_OBJECT_SCHEMA_STRING


def test_schema_hashability(SimpleObject, ComplexObject):
    fields = SimpleObject.__descriptors__.fetch()
    simple_schema = Schema(fields=fields.values())
    fields = ComplexObject.__descriptors__.fetch("datafields")
    validators = [
        d.method for d in ComplexObject.__descriptors__.fetch("datavalidators").values()
    ]
    complex_schema = DataSchema(fields=fields.values(), validators=validators)
    dict(simple_schema=SimpleObject, complex_schema=ComplexObject)


def test_schema_operations(SimpleObject, DerivedObject, OtherDerivedObject):
    fields = SimpleObject.__descriptors__.fetch()
    simple_schema = Schema(fields=fields.values())
    fields = DerivedObject.__descriptors__.fetch()
    derived_schema = Schema(fields=fields.values())
    fields = OtherDerivedObject.__descriptors__.fetch(recursive=True)
    other_derived_schema = Schema(fields=fields.values())
    simple_and_derived = simple_schema & derived_schema
    assert not simple_and_derived.fields
    with pytest.raises(ValueError):
        simple_schema | derived_schema
    assert simple_schema >= simple_and_derived
    assert simple_and_derived <= simple_schema
    assert simple_and_derived <= derived_schema
    assert other_derived_schema >= simple_schema
    assert other_derived_schema | simple_schema == other_derived_schema
    assert simple_schema + other_derived_schema == other_derived_schema


def test_schema_to_pydantic_model(SimpleObject):
    fields = SimpleObject.__descriptors__.fetch()
    schema = Schema(fields=fields.values())
    model_cls = schema.model_class()
    model = model_cls(x=0)
    assert model.x == 0
    assert model.print_hello is False
    with pytest.raises(ValidationError):
        model_cls(x="x")


def test_validators_and_parsers(ComplexObject):
    fields = ComplexObject.__descriptors__.fetch("datafields")
    validators = [
        d.method for d in ComplexObject.__descriptors__.fetch("datavalidators").values()
    ]
    schema = DataSchema(fields=fields.values(), validators=validators)
    model_cls = schema.model_class(namespace=ComplexObject)
    obj = model_cls(parsed=10, twice_validated=5, collection=[0, 1])
    assert obj.parsed == hash(10)
    assert obj.twice_validated == 5
    assert obj.collection == [0, 1]
    with pytest.raises(ValidationError) as exc_info:
        obj = model_cls(parsed=0, twice_validated=15, collection=[0, 1])
        assert exc_info.errors()[0]["msg"] == "Value must be less than 10"
    with pytest.raises(ValidationError):
        obj = model_cls(parsed=10, twice_validated=-10, collection=[0, 1])
    with pytest.raises(ValidationError):
        obj = model_cls(parsed=10, twice_validated=5, collection=["x"])
    # Tests root validation
    with pytest.raises(ValidationError):
        obj = model_cls(parsed=0, twice_validated=5, collection=[0, 1])

    obj = model_cls(parsed="whatever", twice_validated=5, collection=[])
    assert isinstance(obj.parsed, int)
    assert obj.optional is None


def test_single_field_model(ComplexObject):
    field: DataField = ComplexObject.twice_validated
    dataspec = field.dataspec()
    model_cls = dataspec.model_class(namespace=ComplexObject)
    assert model_cls(data=5)
    with pytest.raises(ValidationError):
        model_cls(data=15)
    with pytest.raises(ValidationError):
        model_cls(data=-5)
