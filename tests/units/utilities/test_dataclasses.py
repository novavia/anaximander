import dataclasses
import pydantic
from anaximander.utilities.nxdataclasses import (
    Validated,
    field_info,
    is_frozen,
    pydantic_model_class,
    validate_data,
    is_frozen,
    dictlike,
)

from dataclasses import dataclass

from pydantic import BaseModel
import pytest


def test_validated():
    @dataclass
    class Data(Validated):
        x: int
        y: int = 0

    assert issubclass(Data.__pydantic_model__, BaseModel)
    with pytest.raises(pydantic.ValidationError):
        data = Data("Hello, World!")
    data = Data(0)
    assert data.x == data.y == 0
    data.x = "Hello, World!"
    with pytest.raises(pydantic.ValidationError):
        validate_data(data)


def test_dictlike():
    @dictlike
    @dataclass(frozen=True)
    class Data(Validated):
        x: int
        y: int = 0

    assert is_frozen(Data)
    data = Data(x=0)
    assert dataclasses.asdict(data) == data
    assert data.x == data["x"] == 0

    with pytest.raises(TypeError):

        @dictlike
        @dataclass
        class Data(Validated):
            x: int
            y: int = 0


def test_model_export():
    @dataclass
    class Data(Validated):
        x: int
        y: int = 0

    fields = {f.name: field_info(f) for f in dataclasses.fields(Data)}
    model_class = pydantic_model_class(
        "Data", fields=fields, annotations=Data.__annotations__
    )

    assert list(model_class.__fields__) == ["x", "y"]
    assert is_frozen(model_class.__dictlike_dataclass__)
    model = model_class(x=0)
    assert model.dictlike_dataclass() == {"x": 0, "y": 0}
