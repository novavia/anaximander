from dataclasses import dataclass
from enum import Enum

import pytest
import runtype
from anaximander.aml.meta import (
    Metadescriptor,
    Prototype,
    data,
    model,
)


class C:
    x: int = Metadescriptor()  # type: ignore


class RestrictedMetadescriptor(Metadescriptor):
    __reserved_names__ = ["x"]


class Color(Enum):
    red = "red"
    green = "green"
    blue = "blue"


@dataclass
class DC:
    x: int


def test_metadescriptor_set_name():
    assert Metadescriptor.__reserved_names__ == set()
    assert RestrictedMetadescriptor.__reserved_names__ == {"x"}
    assert C.x.name == "x"  # type: ignore
    with pytest.raises(ValueError):

        class D:
            x: int = RestrictedMetadescriptor()  # type: ignore


def test_prototype_metadescriptors():
    class A(metaclass=Prototype):
        a: int = RestrictedMetadescriptor()  # type: ignore

    class B(A):
        b: int = Metadescriptor()  # type: ignore

    class C(B):
        c: int = Metadescriptor()  # type: ignore

    assert A.metadescriptors() == {"a": A.a}
    assert B.metadescriptors() == {"a": A.a, "b": B.b}
    assert C.metadescriptors() == {"a": A.a, "b": B.b, "c": C.c}
    assert A.metadescriptors(RestrictedMetadescriptor) == {"a": A.a}
    assert B.metadescriptors(RestrictedMetadescriptor, inherited=False) == {}
    assert C.metadescriptors(inherited=False) == {"c": C.c}


def test_data():
    assert runtype.isa(0, data)
    assert runtype.isa(0.0, data)
    assert not runtype.isa([0, 1], data)
    assert not runtype.isa(C(), data)
    assert runtype.issubclass(int, data)
    assert runtype.issubclass(Color, data)
    assert not runtype.issubclass(C, data)


def test_model():
    assert not runtype.isa(C(), model)
    assert runtype.issubclass(DC, model)
