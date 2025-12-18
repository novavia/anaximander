"""Tests for KwargMap behavior and Config integration with OmegaConf."""
from dataclasses import dataclass

import omegaconf
import pytest

from anaximander.utils import Config, KwargMap


def test_kwarg_map():
    """Validate KwargMap contextual lookup, '_' shorthand, ordering, sorting, and subselection.

    Asserts:
        - Mapping semantics and string rendering.
        - Type enforcement on keys.
        - Context resolution from mapping and locals().
        - '_' shorthand for attribute lookup.
        - Deletion behavior and error on missing keys.
        - Sorting behavior and KeyError on unknown keys.
        - __call__ subselection returns ordered subset.
    """
    km_0 = KwargMap(a=1, b=2)
    assert dict(km_0) == {"a": 1, "b": 2}
    assert str(km_0) == "a=1, b=2"

    # only string keys
    with pytest.raises(TypeError):
        km_0[1] = 2  # type: ignore

    # pass a context (a mapping)
    km_1 = KwargMap(km_0, a=".a")
    assert km_1["a"] == 1
    assert str(km_1) == "a=1"

    # pass no context (defaults to local)
    km_2 = KwargMap(a=".km_0")
    assert km_2["a"] == km_0

    # Now use an object as context
    @dataclass
    class C:
        x: int
        y: int

    # test the "_" key
    km_3 = KwargMap(C(1, 2), _=["x", "y"])
    assert km_3["x"] == 1
    assert km_3["y"] == 2

    # test key deletion
    del km_3["x"]
    assert str(km_3) == "y=2"
    with pytest.raises(KeyError):
        del km_3["x"]

    # test insertion, including redundant keys
    km_3["y"] = ".y"
    assert km_3.data == {"y": ".y"}
    km_3["_"] = ["x"]
    assert str(km_3) == "y=2, x=1"

    # test sorting
    km_3.sort()
    assert str(km_3) == "x=1, y=2"
    km_3.sort(["y", "x"])
    assert str(km_3) == "y=2, x=1"
    km_3.sort(["y", "x"], reverse=True)
    assert str(km_3) == "x=1, y=2"
    with pytest.raises(KeyError):
        km_3.sort(["x", "y", "z"])

    # test calling
    km_4 = km_3("x")
    assert str(km_4) == "x=1"


@pytest.mark.bugger
def test_config():
    """Ensure Config subclasses integrate with OmegaConf via the omegaconf property.

    Asserts:
        - Property returns an OmegaConf DictConfig for a Config subclass instance.
    """
    class MyConfig(Config):
        x: int
        y: int = 0

    mc = MyConfig(x=1)
    conf = mc.omegaconf
    assert isinstance(conf, omegaconf.DictConfig)
