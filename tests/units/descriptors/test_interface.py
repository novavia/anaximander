import pytest

from pydantic import ValidationError

from anaximander.descriptors import data, metadata
from anaximander.descriptors.fields import DataField


@pytest.fixture(scope="module")
def Object():
    class Object:
        x: int = data(gt=0)
        y: str = metadata(typespec=True)

    return Object


def test_input_validation():
    with pytest.raises(ValidationError):
        data(default_factory="Eat this!")


def test_data_interface(Object):
    x = Object.x
    assert isinstance(x, DataField)
    assert x.name == "x"
    assert x.type_ is int
    assert x.pydantic_validators == {"gt": 0}
    assert Object.y.typespec
