import pandas as pd
import pytest

from anaximander.dataobjects import Column, Data


def test_column(PositiveFloat, WeirdInteger):
    s0 = [1, 2, 3]
    s1 = [0, 1, 2]
    s2 = [None, 1]
    s3 = ["a", "b", "c"]
    s4 = [1, 2, 1e10]
    PositiveFloatColumn = Column.subtype(datatype=PositiveFloat)
    S0 = PositiveFloatColumn(s0)
    pd.Series(s0).equals(S0.data)
    with pytest.raises(ValueError):
        PositiveFloatColumn(s1)
    # This raises a ValueError and not a TypeError because pandas preprocesses
    # s2 into [NaN, 1.0] and float accepts NaN as an argument. Not sure whether
    # this is the desired behavior or not, so leaving this open-ended for now.
    with pytest.raises(ValueError):
        PositiveFloatColumn(s2)
    # This on the other hand unambiguously raises a TypeError.
    with pytest.raises(TypeError):
        PositiveFloatColumn(s3)
    with pytest.raises(ValueError):
        PositiveFloatColumn(s4)
    WeirdIntegerColumn = Column[WeirdInteger]
    S0_ = WeirdIntegerColumn(s0)
    assert list(S0_.data) == s0
    S3 = WeirdIntegerColumn(data=s3)
    assert all(isinstance(i, int) for i in S3.data)

    # Test slices
    assert S3._row_from_iloc(0) == WeirdInteger("a")
    assert S3._slice_from_iloc(0, 3) == S3
    assert S3._slice_from_iloc(0, 1) == WeirdIntegerColumn(data=["a"])
    assert next(S3.values()) == WeirdInteger("a")
