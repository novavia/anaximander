from dataclasses import dataclass
from enum import Enum
from types import NoneType
from typing import TypedDict

import pytest
import runtype

from anaximander.aml.meta import (
    DataObjectABC,
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


class ColorDict(TypedDict):
    red: int
    green: int
    blue: int


class RainbowDict(ColorDict):
    yellow: int
    orange: int
    indigo: int
    violet: int


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


def test_dataobject_subtype():
    assert DataObjectABC.__is_subtype__(int)
    assert DataObjectABC.__is_subtype__(list[int])
    assert DataObjectABC.__is_subtype__(Color)
    assert DataObjectABC.__is_subtype__(DC)
    assert DataObjectABC.__is_subtype__(list[DC])
    assert DataObjectABC.__is_subtype__(dict[str, list[DC]])
    assert DataObjectABC.__is_subtype__(ColorDict)
    assert DataObjectABC.__is_subtype__(RainbowDict)
    assert not DataObjectABC.__is_subtype__(NoneType)
    assert not DataObjectABC.__is_subtype__(C)
    assert not DataObjectABC.__is_subtype__(list)
    assert not DataObjectABC.__is_subtype__(list[C])
    # with data as *super
    assert DataObjectABC.__is_subtype__(int, data)
    assert DataObjectABC.__is_subtype__(list[int], data)
    assert DataObjectABC.__is_subtype__(Color, data)
    assert DataObjectABC.__is_subtype__(dict[str, list[int]], data)
    assert DataObjectABC.__is_subtype__(ColorDict, data)
    assert DataObjectABC.__is_subtype__(RainbowDict, data)
    assert not DataObjectABC.__is_subtype__(DC, data)
