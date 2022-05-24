import pytest

from anaximander.descriptors.datatypes import nxdata


def test_data(WeirdInteger, PositiveFloat, NullablePositiveFloat):
    n = WeirdInteger(1)
    assert n.data == n._data == 1
    assert isinstance(n, nxdata)
    assert WeirdInteger(data=1) == n
    assert WeirdInteger("xyz").data == abs(hash("xyz"))
    assert WeirdInteger(-1, validate=False).data == -1
    with pytest.raises(ValueError):
        WeirdInteger(-1)
    with pytest.raises(TypeError):
        WeirdInteger("xyz", parse=False)
    assert WeirdInteger(1, parse=False, validate=False).data == 1
    assert WeirdInteger("1").data == 1
    # An invalid instance, due to all checks being turned off
    assert WeirdInteger("1", parse=False, conform=False, validate=False).data == "1"
    assert PositiveFloat.dataspec.pytype is float
    f = PositiveFloat(5)
    assert f.data == 5
    with pytest.raises(ValueError):
        PositiveFloat(0)
    # Tests that pydantic validators in the namespace are incorporated (le = 1e9)
    with pytest.raises(ValueError):
        PositiveFloat(1e10)
    # Tests that types cannot be extended with contradicting validators
    with pytest.raises(ValueError):
        PositiveFloat.dataspec.extend(le=1e10)
    with pytest.raises(ValueError):

        class ContradictoryPositiveFloat(PositiveFloat):
            le = 1e10

    with pytest.raises(TypeError):
        PositiveFloat(None)
    NullablePositiveFloat(None).data is None
    with pytest.raises(ValueError):
        NullablePositiveFloat(0)
