from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Optional, get_type_hints

from pydantic import BaseModel

from anaximander.aml.metadescriptors.datatypes import dataobject, data, model


@dataclass
class C:
    x: int


class D(BaseModel):
    x: int


class Color(Enum):
    red = "red"
    green = "green"
    blue = "blue"


class ValidFields:
    a: int
    b: list[int]
    c: dict[str, list[int] | None]
    d: D
    e: "E"
    f: dict[str, C]
    g: Color
    h: Literal[42, "foo", None, Color.red]
    i: Optional[Color | D | list["E"]]


class InvalidFields:
    a: None
    b: Literal[None]
    c: object
    d: Callable


class E(D):
    pass


def test_is_subclass():
    assert issubclass(bool, dataobject)
    assert issubclass(bool, data)
    assert not issubclass(bool, model)

    assert issubclass(C, dataobject)
    assert not issubclass(C, data)
    assert issubclass(C, model)

    assert issubclass(D, dataobject)
    assert not issubclass(D, data)
    assert issubclass(D, model)


def test_validate_hints():
    valid_hints = get_type_hints(ValidFields)
    for hint in valid_hints.values():
        assert dataobject.validate_hint(hint)

    invalid_hints = get_type_hints(InvalidFields)
    for hint in invalid_hints.values():
        assert not dataobject.validate_hint(hint)
