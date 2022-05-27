from dataclasses import dataclass

import pytest

from anaximander.utilities.functions import (
    mandatory_arguments,
    method_arguments,
    namespace,
    process_timer,
    singleton,
    sort_by,
    typeproperty,
    ProcessTimer,
    process_timer,
    print_process_time,
)


def test_mandatory_arguments():
    def f(x, y=0, **kwargs):
        pass

    def g(x, /, *args, z):
        pass

    assert mandatory_arguments(f) == ["x"]
    assert mandatory_arguments(g) == ["x", "z"]


def test_singleton():
    @singleton
    class ONE:
        pass

    assert ONE is ONE._instance

    class AdvancedType(type):
        pass

    @singleton(42)
    @dataclass
    class AdvancedOne(metaclass=AdvancedType):
        universal_answer: int

    assert AdvancedOne is AdvancedOne._instance
    assert AdvancedOne.universal_answer == 42

    class DerivedOne(type(ONE)):
        pass

    assert DerivedOne() is DerivedOne._instance

    @singleton(42)
    @dataclass
    class OtherDerivedOne(type(ONE)):
        universal_answer: int

    assert OtherDerivedOne is OtherDerivedOne._instance
    assert OtherDerivedOne.universal_answer == 42
    assert isinstance(OtherDerivedOne, type(ONE))


def test_typeproperty():
    class AbstractObjectType(type):
        @typeproperty
        def name(cls):
            return cls.__name__

    class ConcreteObjectType(AbstractObjectType):
        pass

    class Object(metaclass=ConcreteObjectType):
        @property
        def name(self):
            return type(self).name

    obj = Object()
    assert Object.name == obj.name == "Object"
    with pytest.raises(AttributeError):
        Object.name = "Hello, World!"
    with pytest.raises(AttributeError):
        obj.name = "Hello, World!"
    name = property(lambda self: type(self).name.upper())
    Object.name = name
    assert Object.name == "Object"
    assert obj.name == "OBJECT"
    assert "name" in vars(AbstractObjectType)
    assert "name" not in vars(ConcreteObjectType)


def test_process_timer(capsys):
    with ProcessTimer("Squares") as timer:
        [x * x for x in range(10)]
    assert len(timer.records) == 2
    captured = capsys.readouterr()
    assert captured.out.startswith("Process time for Squares")

    with process_timer("Sequence") as timer:
        [x * x for x in range(10)]
        timer.punch()
        [x * x * x for x in range(10)]
    assert len(timer.records) == 3
    prints = capsys.readouterr().out.split("\n")[:-1]
    assert prints[-1].startswith("Step 2")

    @print_process_time
    def function():
        [x * x for x in range(10_000)]

    function()
    captured = capsys.readouterr()
    assert captured.out.startswith("Process time for function")


def test_namespace():
    x = 0
    assert namespace() == {"x": 0}
    assert namespace(lambda k: k > 0) == {}
    assert namespace(exclude=["x"]) == {}


def test_method_arguments():
    class C:
        def __init__(self, x=0, unpack=True, **kwargs):
            self._settings = method_arguments(exclude=["unpack"], unpack=unpack)
            self.x = 0

    c = C(1)
    assert c._settings == {"x": 1}
    c = C(1, y=1)
    assert c._settings == {"x": 1, "y": 1}
    c = C(unpack=False, y=1)
    assert c._settings == {"x": 0, "kwargs": {"y": 1}}
    c = C(unpack="kwargs", y=1)
    assert c._settings == {"x": 0, "y": 1}


def test_sort_by():
    i0 = ["apple", "mint", "orange", "lemon", "banana"]
    i1 = ["banana", "apple", "orange"]
    s0 = list(sort_by(i0, i1))
    assert s0 == ["banana", "apple", "orange", "mint", "lemon"]
    s1 = list(sort_by(i1, target=i0))
    assert s1 == ["apple", "orange", "banana"]
