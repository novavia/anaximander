from collections.abc import Collection, Mapping, Sequence
from enum import Enum
from io import StringIO
from numbers import Number
from typing import Optional
import pytest

from pydantic import ValidationError

from anaximander.utilities.nxyaml import yaml
from anaximander.utilities.functions import attribute_owner, typeproperty
from anaximander.descriptors import metadata, metaproperty, option, metamethod
from anaximander.descriptors.fields import MetadataField
from anaximander.descriptors.attributes import Attribute
from anaximander.descriptors.typespec import typespec
from anaximander.meta.arche import Arche
from anaximander.meta.archetype import Archetype, archetype, metamorph
from anaximander.meta.nxobject import nxobject, nxtrait
from anaximander.meta.nxtype import nxtype
from anaximander.meta.trait import trait
from anaximander.meta import Object


@pytest.fixture(scope="module")
def TestObject():
    @archetype
    class TestObject(Object.Base):
        param: int = metadata(typespec=True)
        owner: Optional[str] = metadata()

        @metaproperty("cool")
        def greater_than_100(param):
            return param >= 100

        @metaproperty("cool")
        def negative(param):
            return param < 0

        @metaproperty("lame")
        def too_big(param):
            return param > 1_000_000

        @metamethod
        def square(cls, param):
            @classmethod
            def method(cls_):
                return param**2

            return method

        @metamethod
        def cube(cls, param, **metadata):
            def owner_method(self):
                return param**3

            def no_owner_method(self):
                return NotImplemented

            if (owner := metadata.get("owner")) is None:
                return no_owner_method
            else:
                return owner_method

    return TestObject


@pytest.fixture(scope="module")
def TestObject_0(TestObject):
    class TestObject_0(TestObject.Base):
        param = 0

    return TestObject_0


@pytest.fixture(scope="module")
def TestObject_1(TestObject):
    class TestObject_1(TestObject.Base, param=1):
        pass

    return TestObject_1


@pytest.fixture(scope="module")
def TestObject_1b(TestObject, TestObject_1):
    class TestObject_1b(TestObject.Base, param=1, overtype=True):
        pass

    return TestObject_1b


@pytest.fixture(scope="module")
def LameObject(TestObject):
    @trait("lame")
    class LameObject(TestObject.Base):
        @metaproperty
        def lame(**metadata):
            return True

    return LameObject


class Owner(Enum):
    ALICE = 1
    BOB = 2


@pytest.fixture(scope="module")
def DerivedObject(TestObject, CoolObject, LameObject):
    @metamorph
    @archetype
    class DerivedObject(TestObject.Base, metacharacters=["lame"]):
        param: float = metadata(typespec=True)
        owner: Optional[Owner] = metadata()

        @param.validator
        def is_a_number(value):
            """Value must be a number"""
            return isinstance(value, Number)

        @metaproperty("alice")
        def is_owned_by_alice(owner):
            return owner is Owner.ALICE

    return DerivedObject


@pytest.fixture(scope="module")
def DerivedObject_0(DerivedObject):
    class DerivedObject_0(DerivedObject.Base):
        param = 0.0

    return DerivedObject_0


@pytest.fixture(scope="module")
def DumbDerivedObject(TestObject):
    """Inherits from TestObject without redefining param."""

    @archetype
    class DumbDerivedObject(TestObject.Base):
        pass

    return DumbDerivedObject


@pytest.fixture(scope="module")
def NotDerivedObject():
    @archetype
    class NotDerivedObject(Object.Base):
        pass

    return NotDerivedObject


@pytest.fixture(scope="module")
def CoolObject(TestObject):
    @trait("cool")
    class CoolObject(TestObject.Base):
        def __init__(self):
            self.cool_param = self.param * 2

        @property
        def cool(self):
            return True

    return CoolObject


@pytest.fixture(scope="module")
def MyCoolObject(TestObject, CoolObject):
    class MyCoolObject(TestObject.Base, metacharacters=["cool"]):
        param = 0

    return MyCoolObject


@pytest.fixture(scope="module")
def CoolerObject(TestObject, DerivedObject):
    @trait("cooler")
    class CoolerObject(DerivedObject):
        def __init__(self):
            self.cool_param = self.param * 2
            self.cool_param *= 2

        @property
        def cooler(self):
            return True

    return CoolerObject


@pytest.fixture(scope="module")
def TrulyCoolObject(CoolerObject, DerivedObject):
    @trait("cool")
    class TrulyCoolObject(DerivedObject.Base):
        def __init__(self):
            self.cool_param = self.param * 2
            self.cool_param *= 5

        @property
        def cool(self):
            return True

    return TrulyCoolObject


@pytest.fixture(scope="module")
def CoolestObject(CoolerObject):
    @trait("coolest")
    class CoolestObject(CoolerObject):
        @property
        def coolest(self):
            return True

    return CoolestObject


@pytest.fixture(scope="module")
def AliceObject(DerivedObject):
    @trait("alice")
    class AliceObject(DerivedObject):
        pass

    return AliceObject


@pytest.fixture(scope="module")
def DerivedDerivedObject(DerivedObject, AliceObject):
    @metamorph
    @archetype
    class DerivedDerivedObject(DerivedObject.Base, metacharacters=["alice"]):
        # This fixes the object parameter for instances and derived types
        param = 0.0
        owner = Owner.ALICE
        x: float = metadata()

    return DerivedDerivedObject


@pytest.fixture(scope="module")
def CustomObject(DerivedObject):
    class CustomObjectType(DerivedObject.metatype):
        """A custom metaclass"""

        @typeproperty
        def custom(cls):
            return True

    @archetype(metatype=CustomObjectType)
    class CustomObject(DerivedObject):
        """Custom object archetype"""

        print_hello: bool = option(False)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if self.print_hello:
                print("hello")

    return CustomObject


@pytest.fixture(scope="module")
def ObjectWithClassMetadataAttributes():
    @archetype
    class ObjectWithClassMetadataAttributes(Object.Base):
        collection_type: type[Collection] = metadata(typespec=True)

    return ObjectWithClassMetadataAttributes


@pytest.fixture(scope="module")
def ObjectWithAliasedMetadata(TestObject):
    @archetype
    class ObjectWithAliasedMetadata(TestObject):
        x: int = metadata(alias="y")

    return ObjectWithAliasedMetadata


def test_seed_class():
    @archetype
    class Object(nxobject, metaclass=nxtype):
        param: str = metadata(typespec=True)

    assert tuple(Object.metadata.fields) == ("param",)
    assert isinstance(Object.param, MetadataField)
    assert Object.Base.param is None
    assert Object.Base.abstract


def test_new_archetype(TestObject: Archetype):
    assert isinstance(TestObject, Archetype)
    assert TestObject.archetype is TestObject
    assert TestObject.basetype.__name__ == "TestObject"
    assert TestObject.basetype.__base__ is Object.basetype
    assert TestObject.__base__ is Object
    assert set(TestObject.metadata.fields) == {"param", "owner"}
    assert TestObject.metatype.__name__ == "TestObjectType"
    assert TestObject.metatype.__base__ is Object.metatype
    # Archetypes are not writable
    with pytest.raises(AttributeError):
        TestObject.x = 0


def test_custom_archetype(CustomObject: Archetype, capsys):
    assert isinstance(CustomObject, Archetype)
    assert CustomObject.Base.custom
    MyCustomObject = CustomObject.subtype(param=3.14)
    no_print_obj = MyCustomObject()
    assert no_print_obj.custom
    captured = capsys.readouterr()
    assert captured.out == ""
    print_obj = MyCustomObject(print_hello=True)
    assert print_obj.custom
    captured = capsys.readouterr()
    assert captured.out == "hello\n"


def test_Base(TestObject: Archetype):
    class A(TestObject):
        param = 2

    class B(TestObject.Base):
        param = 3

    assert A.mro()[1:] == B.mro()[1:]


def test_metadata(
    TestObject: Archetype,
    DerivedObject: Archetype,
    DumbDerivedObject: Archetype,
    TestObject_0: nxtype,
    TestObject_1: nxtype,
    TestObject_1b: nxtype,
):
    assert isinstance(TestObject.param, MetadataField)
    assert TestObject.param.typespec
    assert isinstance(TestObject.metatype.param, Attribute)
    assert isinstance(TestObject.basetype.param, Attribute)
    assert TestObject.Base.param is None
    # Test declarations
    assert list(TestObject.metadata.fields)[0] == "param"
    assert DumbDerivedObject.metadata == TestObject.metadata
    assert TestObject.metadata != DerivedObject.metadata

    # Test setters
    assert TestObject_0.param == 0
    assert TestObject_0().param == 0
    assert TestObject_1.param == TestObject_1b.param == 1
    test_meta = TestObject_0.metadata
    assert test_meta.param == test_meta["param"] == 0
    with pytest.raises(ValidationError):

        class TestObjectFloat(TestObject.Base):
            param = "x"

    # Test class-level immutability
    with pytest.raises(AttributeError):
        TestObject_0.param = 0

    assert DerivedObject.basetype.abstract
    with pytest.raises(AttributeError):
        # This fails because the base type is already initialized
        DerivedObject.basetype.param = 0

    # Test subclassing immutability
    with pytest.raises(ValidationError):

        class X(TestObject_0, param=1):
            pass

    with pytest.raises(ValidationError):

        class X(TestObject_0):
            param = 1

    # Test instance-level immutability
    with pytest.raises(AttributeError):
        TestObject_0().param = 0

    # Test parameter fixing
    @archetype
    class FixedObject(TestObject, param=2):
        pass

    assert FixedObject.param == 2
    with pytest.raises(AttributeError):
        FixedObject.param = 3

    MyFixedObject = FixedObject.subtype()
    assert MyFixedObject.param == 2

    # Verify correct descriptor types
    assert isinstance(FixedObject.metatype.param, Attribute)
    owner = attribute_owner(MyFixedObject, "param")
    assert owner is TestObject.basetype
    assert isinstance(TestObject.basetype.param, Attribute)

    with pytest.raises(AttributeError):
        MyFixedObject.param = 3

    # Cannot reset a parameter, even if it has been fixed by FixedObject's
    # base type and is hence not listd as a parameter.
    with pytest.raises(ValidationError):

        class MyOtherFixedObject(FixedObject):
            param = 3

    with pytest.raises(ValidationError):

        class MyOtherFixedObject(FixedObject, param=3):
            pass

    with pytest.raises(ValidationError):
        FixedObject.subtype(param=3)


def test_spec_registration_and_retrieval(
    TestObject: Archetype,
    TestObject_0: nxtype,
    DerivedObject: Archetype,
    DerivedObject_0: nxtype,
):
    assert TestObject.retrieve_type(param=0) is TestObject_0
    assert DerivedObject.retrieve_type(param=0.0) is DerivedObject_0
    with pytest.raises(KeyError):
        TestObject.retrieve_type(param=5)
    MyObject = TestObject.subtype(type_name="MyObject")
    assert issubclass(MyObject, TestObject)


def test_key_registration_and_retrieval(
    TestObject: Archetype,
    TestObject_0: nxtype,
    DerivedObject: Archetype,
    DerivedObject_0: nxtype,
):
    assert TestObject.retrieve_type_from_key(0) is TestObject_0
    with pytest.raises(KeyError):
        TestObject.retrieve_type_from_key(5)
    assert DerivedObject[0] is DerivedObject_0


def test_instance_parameter_setting(TestObject: Archetype, TestObject_0: nxtype):
    obj = TestObject(param=0)
    assert type(obj) is TestObject_0
    assert obj.param == 0
    # This works too even though there is redefinition of metadata, because the
    # setting is the same
    obj = TestObject_0(param=0)
    # But this is invalid
    with pytest.raises(ValidationError):
        TestObject_0(param=1)


def test_issubclass_relationships(
    TestObject: Archetype,
    TestObject_0: nxtype,
    DerivedObject: Archetype,
    DerivedObject_0: nxtype,
    TestObject_1: nxtype,
    NotDerivedObject: Archetype,
    CoolObject: nxtype,
):
    assert issubclass(TestObject_0, TestObject)
    assert issubclass(DerivedObject, TestObject)
    assert issubclass(DerivedObject_0, TestObject)
    assert not issubclass(TestObject_1, TestObject_0)
    assert not issubclass(NotDerivedObject, TestObject)
    assert issubclass(TestObject, Object)
    assert issubclass(NotDerivedObject, Object)
    assert issubclass(CoolObject, TestObject)
    assert issubclass(TestObject, Arche)
    assert not issubclass(TestObject_0, CoolObject)
    assert issubclass(TestObject_0, Object)


def test_trait(
    TestObject,
    CoolObject,
    MyCoolObject,
    DerivedObject,
    CoolerObject,
    CoolestObject,
    TrulyCoolObject,
    DerivedDerivedObject,
    LameObject,
):
    assert isinstance(CoolObject, nxtrait)
    assert TestObject.metacharacters == ("cool", "lame")
    assert DerivedObject.metacharacters == (
        "lame",
        "cooler",
        "coolest",
        "cool",
        "alice",
    )
    assert DerivedDerivedObject.metacharacters == (
        "lame",
        "cooler",
        "coolest",
        "cool",
        "alice",
    )
    assert DerivedDerivedObject.basetype.metacharacters == {"lame", "alice"}
    assert DerivedDerivedObject.Base.metacharacters == {"lame", "alice"}
    assert DerivedDerivedObject.basetype.typespec.metacharacters == set()
    assert DerivedDerivedObject.Base.typespec.metacharacters == set()
    assert TestObject.traits == {"lame": LameObject, "cool": CoolObject}

    assert MyCoolObject.traits == (CoolObject,)

    my_cool_object = MyCoolObject()
    assert my_cool_object.param == 0
    assert isinstance(my_cool_object, CoolObject)
    assert my_cool_object.metacharacters == {"cool"}
    # Tests CoolObject's __init__ method
    assert my_cool_object.cool_param == 0

    # Fails because trait cannot decorate concrete types
    with pytest.raises(TypeError):

        @trait("cooler")
        class CoolerObject(MyCoolObject):
            pass

    class MyCoolerObject(DerivedObject, metacharacters=["cooler", "cool"]):
        param = 5.0

    assert MyCoolerObject.traits == (TrulyCoolObject, CoolerObject, LameObject)
    assert MyCoolerObject.metacharacters == {"lame", "cooler", "cool"}
    assert MyCoolerObject.typespec.metacharacters == {"cooler", "cool"}
    # Tests for metaproperty declared by trait
    assert MyCoolerObject.lame
    assert [c.__name__ for c in MyCoolerObject.mro()] == [
        "MyCoolerObject",
        "TrulyCoolObject",
        "CoolerObject",
        "DerivedObject",
        "LameObject",
        "TestObject",
        "Object",
        "nxobject",
        "object",
    ]

    my_cooler_object = MyCoolerObject()
    # Tests that the object goes through CoolObject, then CoolerObject's __init__ methods
    assert my_cooler_object.cool_param == 50.0
    # Tests for metaproperty declared by trait
    assert my_cooler_object.lame

    class MyCoolestObject(
        DerivedObject.Base, metacharacters=["coolest", "cooler", "cool"]
    ):
        param = 0.0

    assert MyCoolestObject.traits == (
        TrulyCoolObject,
        CoolestObject,
        CoolerObject,
        LameObject,
    )
    assert MyCoolestObject.metacharacters == {"lame", "coolest", "cooler", "cool"}
    assert MyCoolestObject.typespec.metacharacters == {"coolest", "cooler", "cool"}


def test_metaproperties(
    TestObject,
    CoolObject,
    MyCoolObject,
    DerivedObject,
    CoolerObject,
    CoolestObject,
    TrulyCoolObject,
    DerivedDerivedObject,
    LameObject,
):
    # We first test on subtypes
    MySoSoObject = TestObject.subtype(param=50)
    MyCoolObject = TestObject.subtype(param=150)
    MyOtherCoolObject = TestObject.subtype(param=-10)
    MyCoolAndLameObject = TestObject.subtype(param=2_000_000)

    assert not MySoSoObject.metacharacters
    assert MyCoolObject.metacharacters == {"cool"}
    assert MyOtherCoolObject.metacharacters == {"cool"}
    # Even though the lame metaproperty is checked, MyCoolAndLameObject
    # is metamorphed to the DerivedObject archetype and does not show
    # the lame trait in its typespec
    assert MyCoolAndLameObject.metacharacters == {"lame", "cool"}
    assert MyCoolAndLameObject.typespec.metacharacters == {"cool"}
    assert MyCoolAndLameObject.archetype is DerivedObject

    # Now we test on instances
    my_so_so_object = TestObject(param=50)
    my_cool_object = TestObject(param=250)
    my_cool_and_lame_object = TestObject(param=2_000_001)

    assert not type(my_so_so_object).metacharacters
    assert type(my_cool_object).metacharacters == {"cool"}
    assert type(my_cool_object).typespec.metacharacters == {"cool"}
    assert my_cool_object.metacharacters == {"cool"}
    assert my_cool_and_lame_object.archetype is DerivedObject


def test_object_metadata(
    TestObject, TestObject_0, DerivedObject, DerivedObject_0, DerivedDerivedObject
):
    assert isinstance(TestObject.owner, MetadataField)
    assert isinstance(TestObject.metatype.owner, Attribute)
    assert isinstance(TestObject.basetype.owner, Attribute)
    obj = TestObject_0()
    assert obj.metadata == {"param": 0, "owner": None}
    assert obj.owner is None
    with pytest.raises(AttributeError):
        obj.owner = "Alice"
    obj = TestObject_0(owner="Alice")
    assert obj.owner == "Alice"
    with pytest.raises(TypeError):
        TestObject_0(owner="Alice", whatever="whatever")
    obj = DerivedObject_0()
    assert obj.owner is None
    with pytest.raises(AttributeError):
        obj.owner = Owner.ALICE
    with pytest.raises(ValidationError):
        DerivedObject_0(owner="Alice")

    # These both fail because parameters cannot be set on an nxtype post initialization
    with pytest.raises(AttributeError):
        DerivedObject_0.owner = "Alice"
    with pytest.raises(AttributeError):
        DerivedObject_0.owner = Owner.BOB

    # This switches the archetype (and hence the type) because there
    # is a metaproperty + metamorphism that interprets Alice ownership
    # as a DerivedObject with 'alice' metacharacter, hence a DerivedDerivedObject
    obj = DerivedObject_0(owner=Owner.ALICE)
    assert obj.owner == Owner.ALICE
    assert obj.archetype == DerivedDerivedObject

    class MyDerivedObject(DerivedObject, param=1.1):
        owner = Owner.BOB

    obj = MyDerivedObject()
    assert obj.owner == Owner.BOB
    with pytest.raises(AttributeError):
        obj.owner = Owner.ALICE

    with pytest.raises(ValidationError):  # Wrong parameter type set on a class

        class MyOtherDerivedObject(DerivedObject):
            owner = "Bob"

    class MyDerivedDerivedObject(DerivedDerivedObject, x=0):
        pass

    MyDerivedDerivedObject.x == 0
    MyDerivedDerivedObject().x == 0
    assert isinstance(DerivedDerivedObject.metatype.owner, Attribute)
    owner_type = attribute_owner(MyDerivedDerivedObject, "owner")
    assert owner_type is DerivedObject.basetype
    assert isinstance(DerivedObject.basetype.__dict__["owner"], Attribute)

    obj = MyDerivedDerivedObject()
    assert obj.owner == Owner.ALICE
    with pytest.raises(AttributeError):
        MyDerivedDerivedObject.owner = Owner.BOB
    with pytest.raises(AttributeError):
        obj.owner = Owner.BOB


def test_init_kwargs(TestObject, ObjectWithAliasedMetadata):
    assert TestObject.__kwargs__ == TestObject.__type_kwargs__ == {"param", "owner"}
    assert ObjectWithAliasedMetadata.__kwargs__ == {"param", "owner", "x"}
    assert ObjectWithAliasedMetadata.__type_kwargs__ == {"param", "owner", "x"}


def test_metamethods(TestObject):
    obj = TestObject(param=10)
    assert obj.square() == 100
    assert obj.cube() == NotImplemented
    # This is a classmethod
    assert TestObject[10].square() == 100
    # This is a regular instance method
    with pytest.raises(TypeError):
        TestObject[10].cube()


def test_overwriting_metadata_with_subclass(ObjectWithClassMetadataAttributes):
    arc: Archetype = ObjectWithClassMetadataAttributes
    cls = arc.subtype(collection_type=Mapping)
    with pytest.raises(ValidationError):

        class C(cls):
            collection_type = Sequence

    class D(cls):
        collection_type = dict

    assert D.collection_type is dict
    assert cls(collection_type=Mapping).collection_type is Mapping
    assert cls(collection_type=dict).collection_type is dict


YAML_STRING = """!nxtype
type_name: MyCoolObject
archetype: TestObject
metacharacters:
- cool
metadata:
  param: 0
"""


def test_representations(TestObject, TestObject_1, CoolObject, MyCoolObject):
    assert repr(TestObject) == "<archetype:TestObject>"
    assert repr(TestObject_1) == "<nxtype:TestObject_1>"
    assert repr(TestObject_1()) == "<nxobject:TestObject_1>"
    assert repr(CoolObject) == "<nxtrait:CoolObject>"
    assert repr(TestObject[10]) == "<nxtype:TestObject|param=10>"
    assert (
        repr(TestObject.subtype("cool", param=10))
        == "<nxtype:TestObject|cool|param=10>"
    )
    assert str(MyCoolObject) == YAML_STRING


# =========================================================================== #
#                        Tests for type specifications                        #
# =========================================================================== #


YAML_SPEC = """!typespec
archetype: TestObject
metacharacters:
- cool
metadata:
  param: 0
"""


def test_spec_representations(TestObject):
    spec_a = typespec(TestObject, "cool", param=0)
    assert repr(spec_a) == "<typespec:TestObject|cool|param=0>"
    assert str(spec_a) == YAML_SPEC


@pytest.mark.xfail
def test_yaml_roundtrip(TestObject):
    spec_a = typespec(TestObject, "cool", param=0)
    string = StringIO()
    yaml.dump(spec_a, string)
    assert yaml.load(string.getvalue()) == spec_a


def test_spec_equality(TestObject, DerivedObject):
    spec_a = typespec(TestObject, "cool", param=0)
    spec_b = typespec(TestObject, "cool", param=0)
    assert spec_a == spec_b
    spec_c = typespec(TestObject, "cool", param=1)
    assert spec_a != spec_c
    spec_d = typespec(DerivedObject, "cool", param=0)
    assert spec_a != spec_d


def test_validate(TestObject, DerivedObject, CoolObject):
    spec_a = typespec(TestObject, "cool", param=0)
    assert spec_a.validate()
    spec_b = typespec(TestObject, "cool", whatever=0)
    assert not spec_b.validate()
    spec_c = typespec(DerivedObject, "cool", param=0)
    assert spec_c.validate()


def test_ordering(TestObject, DerivedObject, NotDerivedObject):
    spec_a = typespec(TestObject, param=0)
    spec_b = typespec(TestObject, "cool", param=0)
    spec_c = typespec(TestObject, param=1)
    spec_d = typespec(DerivedObject, "cool", param=0, other_param=1)
    spec_e = typespec(DerivedObject, "cool", param=1, other_param=1)
    spec_f = typespec(NotDerivedObject, param=0)
    assert spec_a <= spec_b
    assert spec_b >= spec_a
    assert spec_a < spec_b
    assert spec_b > spec_a
    assert not spec_a >= spec_c
    assert not spec_a <= spec_c
    assert spec_d >= spec_a
    assert spec_c <= spec_e
    assert not spec_f >= spec_a
    assert not spec_f <= spec_a


def test_operations(TestObject, DerivedObject, NotDerivedObject):
    spec_a = typespec(TestObject, param=0)
    spec_b = typespec(TestObject, "cool", param=0)
    spec_c = typespec(TestObject, param=1)
    spec_d = typespec(DerivedObject, "cool", param=0, other_param=1)
    spec_e = typespec(DerivedObject, "cool", param=1, other_param=1)
    spec_f = typespec(NotDerivedObject, param=0)
    assert spec_a & spec_b == spec_a
    assert spec_a | spec_b == spec_b
    assert spec_a & spec_c == typespec(TestObject)
    assert spec_d & spec_a == spec_a
    assert spec_c | spec_e == spec_e
    with pytest.raises(TypeError):
        assert spec_a & spec_f
