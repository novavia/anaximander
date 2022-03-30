import datetime as dt
from typing import Type
from attr import validate
import pytest
from pydantic import ValidationError

from anaximander.descriptors.dataspec import PydanticValidatorMap, DataSpec


def test_validator_map():
    m0 = dict(gt=0, le=10)
    m1 = dict(multiple_of=5)
    m2 = dict(gt=0, le=5)
    m3 = dict(gt=1, le=5, multiple_of=5)
    m4 = dict(gt=-1, le=5)
    m5 = dict(multiple_of=25)
    m6 = dict(multiple_of=3)
    bad_input_1 = dict(a=0)
    bad_input_2 = dict(min_length=3.5)

    M0 = PydanticValidatorMap(m0)
    M1 = PydanticValidatorMap(**m1)
    M2 = PydanticValidatorMap(m2)
    M3 = PydanticValidatorMap(m3)
    M4 = PydanticValidatorMap(m4)
    M5 = PydanticValidatorMap(m5)
    M6 = PydanticValidatorMap(m6)

    with pytest.raises(TypeError):
        PydanticValidatorMap(bad_input_1)
    with pytest.raises(TypeError):
        PydanticValidatorMap(bad_input_2)

    assert M0["gt"] == 0
    assert M2 > M0
    assert not M1 > M0
    assert M3 >= M1
    assert M2 <= M3
    assert not M0 < M4
    assert M5 > M1
    assert not M6 < M1


def test_data_spec():
    dataspec_0 = DataSpec(int, parser=lambda c, v: hash(v))
    dataspec_1 = dataspec_0.extend(ge=0, le=0)
    dataspec_2 = DataSpec(int)
    dataspec_3 = dataspec_2.extend(nullable=True)
    dataspec_4 = dataspec_3.extend(default=0)
    dataspec_5 = dataspec_4.extend(const=True)
    dataspec_6 = DataSpec(dt.datetime, tz=dt.timezone.utc)
    M0 = dataspec_0.model_class()
    M0_ = dataspec_0.model_class(parse=False)
    M1 = dataspec_1.model_class()
    M1_ = dataspec_1.model_class(validate=False)
    M2 = dataspec_2.model_class()
    M3 = dataspec_3.model_class()
    M4 = dataspec_4.model_class()
    M5 = dataspec_5.model_class()
    M6 = dataspec_6.model_class()
    M0(data=0)
    M0(data="x")
    with pytest.raises(ValidationError):
        M0_(data="x")
    M1(data=0)
    with pytest.raises(ValidationError):
        M1(data=1)
    M1_(data="x")
    with pytest.raises(ValidationError):
        M2(data=None)
    assert M3(data=None).data is M3().data is None
    assert M4().data == 0
    assert M5().data == 0
    with pytest.raises(ValidationError):
        M5(data=1)
    with pytest.raises(TypeError):
        dataspec_3.extend(const=True)
    datetime = dt.datetime.now()
    assert datetime.tzinfo is None
    assert M6(data=datetime).data.tzinfo == dt.timezone.utc
    assert M6(data=datetime).data != datetime
    datetime = dt.datetime.now(dt.timezone.utc)
    assert M6(data=datetime).data == datetime
