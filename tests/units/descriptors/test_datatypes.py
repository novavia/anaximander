from typing import Optional, Union, TypedDict
from anaximander.descriptors.datatypes import (
    nxdata,
    nxmodel,
    datatype,
    pyhint,
)

import pytest


@pytest.fixture(scope="session")
def Integer():
    @datatype(pytype=int)
    class Integer:
        pass

    return Integer


@pytest.fixture(scope="session")
def hints(Integer):
    H0 = int
    H1 = Optional[Integer]
    H2 = list[float]
    H3 = dict[str, Optional[Integer]]
    H4 = dict[str, Union[int, float]]
    H5 = list
    return H0, H1, H2, H3, H4, H5


@pytest.fixture(scope="session")
def Spec(Integer):
    class Template(TypedDict):
        x: int
        y: dict[str, Optional[Integer]]
        z: Optional[Integer]

    return Template


def test_registration(Integer, hints):
    H0, H1, H2, H3, H4, H5 = hints
    assert issubclass(Integer, nxdata)
    assert issubclass(H0, nxdata)
    assert pyhint(Integer) is int
    assert isinstance(0, nxdata)


def test_fieldtype_from_hint(Integer, hints):
    H0, H1, H2, H3, H4, H5 = hints
    assert nxdata.from_hint(H0) == int
    assert nxdata.from_hint(H1) == Integer
    assert nxdata.from_hint(H2) == list
    assert nxdata.from_hint(H3) == dict
    assert nxdata.from_hint(H5) == list
    with pytest.raises(TypeError):
        nxdata.from_hint(H4)


def test_pyhint(hints):
    H0, H1, H2, H3, H4, H5 = hints
    assert pyhint(H0) == int
    assert pyhint(H1) == Optional[int]
    assert pyhint(H2) == list[float]
    assert pyhint(H3) == dict[str, Optional[int]]
    assert pyhint(H5) == list
    with pytest.raises(TypeError):
        pyhint(H4)


def test_pytypes(hints):
    H0, H1, H2, H3, H4, H5 = hints
    assert nxdata.pytype(H0) == int
    assert nxdata.pytype(H1) == int
    assert nxdata.pytype(H2) == list
    assert nxdata.pytype(H3) == dict


def test_nxdata():
    assert nxdata(0) == 0
    assert nxdata(0, dataspec=int, conform=False) == 0
    assert nxdata("0", dataspec=int, conform=True) == 0
    with pytest.raises(TypeError):
        nxdata("0", dataspec=int, conform=False)
    with pytest.raises(TypeError):
        nxdata(None, dataspec=int)
    with pytest.raises(TypeError):
        nxdata(None, dataspec=int, conform=False)
    assert nxdata(None, dataspec=Optional[int]) is None
    assert nxdata([0, None], dataspec=Optional[list[Optional[int]]]) == [0, None]
    assert nxdata((0, None), dataspec=Optional[list[Optional[int]]]) == [0, None]


def test_nxmodel(Integer, Spec):
    assert issubclass(Spec, nxmodel)
    data = dict(x=0, y={"i": None, "j": 0}, z=1)
    s0 = nxmodel(data, modelspec=Spec)
    assert s0 == data
    assert isinstance(data, nxmodel)
    data = dict(x=0, y={"i": "hi", "j": 0}, z=1)
    with pytest.raises(TypeError):
        nxmodel(data, modelspec=Spec)
