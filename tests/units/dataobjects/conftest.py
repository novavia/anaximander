from datetime import datetime
from typing import ClassVar, Optional

import pandas as pd
import pytest

import anaximander as nx

from anaximander.dataobjects import (
    Integer,
    Float,
    Spec,
    Record,
    Sample,
    Session,
    Journal,
)


@pytest.fixture(scope="package")
def WeirdInteger():
    class WeirdInteger(Integer):
        zero: ClassVar[int] = 0

        @nx.data.parser
        def hash_string(cls, value):
            if isinstance(value, int):
                return value
            elif isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return abs(hash(value))
            raise ValueError

        @nx.data.validator
        def positive(cls, value):
            """Value must be greater than or equal to zero"""
            return value >= cls.zero

    return WeirdInteger


@pytest.fixture(scope="package")
def PositiveFloat():
    class PositiveFloat(Float):
        le = 1e9

        @nx.data.validator
        def strictly_positive(cls, value):
            """Value must be strictly positive"""
            return value > 0

    return PositiveFloat


@pytest.fixture(scope="package")
def NullablePositiveFloat(PositiveFloat):
    class NullablePositiveFloat(PositiveFloat):
        nullable = True

    return NullablePositiveFloat


@pytest.fixture(scope="package")
def Point():
    class Point(Spec):
        x: int = nx.data()
        y: int = nx.data()

    return Point


@pytest.fixture(scope="package")
def XPoint():
    class XPoint(Spec, strict_schema=False):
        x: int = nx.data()
        y: int = nx.data()

    return XPoint


@pytest.fixture(scope="package")
def PositivePoint():
    class PositivePoint(Spec):
        x: int = nx.data(gt=0)
        y: int = nx.data(gt=0)

    return PositivePoint


@pytest.fixture(scope="package")
def SimpleFix():
    """A full record schema with key and timestamp."""

    class SimpleFix(Sample):
        mobile: int = nx.key()
        time: datetime = nx.timestamp()
        x: float = nx.data()
        y: float = nx.data()

    return SimpleFix


@pytest.fixture(scope="package")
def Fix(PositiveFloat):
    """Fixture intended to test embedding of compound fields."""

    class Fix(Sample):
        mobile: int = nx.key()
        time: datetime = nx.timestamp(tz="utc")
        x: float = nx.data()
        y: float = nx.data()
        z: PositiveFloat = nx.data()

    return Fix


@pytest.fixture(scope="package")
def WeirdRecord(WeirdInteger):
    """Fixture intended to test embedding of compound fields with weird validation scenarios."""

    class WeirdRecord(Record):
        ten: ClassVar[int] = 10
        key: WeirdInteger = nx.key()
        i: WeirdInteger = nx.data(ge=10)
        j: WeirdInteger = nx.data()
        inverter: Optional[WeirdInteger] = nx.data()

        @key.validator
        def is_even(cls, value):
            """Value must be even"""
            return value % 2 == 0

        # Here the validators is applied to a WeirdInteger instance
        @key.validator(as_object=True)
        def equal_to_zero_or_greater_than_ten(cls, value):
            return value.data == value.zero or value.data >= cls.ten

        @j.validator
        def greater_than_ten(cls, value):
            """Value must be greater than or equal to 10"""
            return value >= cls.ten

        @inverter.parser
        def invert(cls, value):
            return -value

    return WeirdRecord


@pytest.fixture(scope="package")
def DefaultRecord(WeirdInteger, PositiveFloat):
    """Fixture that tests the use of default alongside compound fields."""

    class DefaultRecord(Record):
        key: WeirdInteger = nx.key()
        x: PositiveFloat = nx.data(1.0)
        y: PositiveFloat = nx.data(PositiveFloat(1.0))

    return DefaultRecord


@pytest.fixture(scope="package")
def TempJournal():
    class TempJournal(Journal):
        """A daily temperature summary."""

        machine_id: int = nx.key()
        day: pd.Period = nx.period(freq="D")
        mean_temp: float = nx.data()

    return TempJournal


@pytest.fixture(scope="package")
def Trip():
    class Trip(Session):
        """A daily temperature summary."""

        mobile: int = nx.key()
        start_time: pd.Timestamp = nx.start_time(tz="UTC")
        end_time: pd.Timestamp = nx.end_time(tz="UTC")

    return Trip
