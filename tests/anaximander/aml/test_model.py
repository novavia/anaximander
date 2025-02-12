from enum import Enum
from typing import Literal, Optional, TypedDict, get_type_hints

import pytest
from annotationlib import Format, get_annotations

from anaximander.aml.meta import compile
from anaximander.aml.model import Model
from anaximander.aml.modeldescriptors import Field, field


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


@compile("test")
class C(Model):
    x: int = field()


class TestModel(Model):
    a: int = field()
    b: dict[str, list[str]] = field()
    c: Color = field()
    d: list[Color] = field()
    e: RainbowDict = field()


class InvalidFieldsModel(Model):
    a: TestModel = field()
    b: dict = field()
    c: Optional[float] = field()
    d: Literal["f", 0] = field()
    e: float | str = field()
    f: str | None = field()


def test_compile():
    assert C.__compilations__ == {"test": {}}
    assert C.metadescriptors(Field) == {"x": C.x}


def test_set_annotation():
    C.__set_type_annotations__()
    assert C.x.annotation == "int"  # type: ignore
    assert C.x.hint is int  # type: ignore
    TestModel.__set_type_annotations__()
    annotations = get_annotations(InvalidFieldsModel, format=Format.STRING)
    superhints = get_type_hints(InvalidFieldsModel)
    for name in ("a", "b", "c", "d", "e", "f"):
        metadescriptor = getattr(InvalidFieldsModel, name)
        annotation = annotations[name]
        hint = superhints[name]
        with pytest.raises(TypeError):
            metadescriptor.__set_type__(InvalidFieldsModel, annotation, hint)
