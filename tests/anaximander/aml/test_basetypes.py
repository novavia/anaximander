import pytest

from anaximander.aml.basetypes import Metadata, is_pydata, is_pydata_type, pydata_runtime_type


def test_is_pydata_type_accepts_nested_types():
    assert is_pydata_type(dict[str, list[tuple[int, ...]]]) is True
    assert is_pydata_type(dict[int, dict[str, float]]) is True
    assert is_pydata_type(dict[str, int] | dict[int, int]) is True
    assert is_pydata_type(list[dict[str, int]]) is True


@pytest.mark.parametrize(
    "tp",
    [
        dict[str | int, int],
        dict[str, dict[int, object]],
        tuple[int, int],
    ],
)
def test_is_pydata_type_rejects_invalid_types(tp):
    assert is_pydata_type(tp) is False

def test_is_pydata_accepts_nested_values():
    value = {"a": 1, "b": ["x", (True, {"c": 2.5})]}
    assert is_pydata(value) is True


@pytest.mark.parametrize("value", [{"a": object()}, {object(): 1}, {"a": {(): 1}}, {"a": 1, 2: 2}])
def test_is_pydata_rejects_invalid_values(value):
    assert is_pydata(value) is False

def test_pydata_runtime_type_extracts_container():
    assert pydata_runtime_type(dict[str, list[int]]) is dict
    assert pydata_runtime_type(list[int]) is list
    assert pydata_runtime_type(tuple[int, ...]) is tuple
    assert pydata_runtime_type(int) is int


def test_metadata_subclasshook_accepts_pydata_types():
    assert issubclass(int, Metadata)
    assert issubclass(dict, Metadata)
