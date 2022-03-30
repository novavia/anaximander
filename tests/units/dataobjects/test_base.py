from datetime import datetime
from enum import Enum, IntEnum
from multiprocessing.sharedctypes import Value
from typing import ClassVar, Type, TypedDict, Optional
from attr import validate

import pandas as pd
from pydantic import BaseModel
import pytest
import pytz

import anaximander as nx
from anaximander.utilities.functions import typeproperty
from anaximander import archetype, metaproperty, metadata, metamorph
from anaximander.descriptors.fields import MetadataField, DataField
from anaximander.descriptors.datatypes import nxdata, nxmodel
from anaximander.descriptors.scope import DataScope

from anaximander.dataobjects.base import (
    DataObject,
    Data,
    Integer,
    Float,
    Column,
    Model,
    Entity,
    Table,
    Spec,
    Record,
    Sample,
    Session,
    Journal,
)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def PositiveFloat():
    class PositiveFloat(Float):
        le = 1e9

        @nx.data.validator
        def strictly_positive(cls, value):
            """Value must be strictly positive"""
            return value > 0

    return PositiveFloat


@pytest.fixture(scope="module")
def NullablePositiveFloat(PositiveFloat):
    class NullablePositiveFloat(PositiveFloat):
        nullable = True

    return NullablePositiveFloat


@pytest.fixture(scope="module")
def Point():
    class Point(Spec):
        x: int = nx.data()
        y: int = nx.data()

    return Point


@pytest.fixture(scope="module")
def XPoint():
    class XPoint(Spec, strict_schema=False):
        x: int = nx.data()
        y: int = nx.data()

    return XPoint


@pytest.fixture(scope="module")
def PositivePoint():
    class PositivePoint(Spec):
        x: int = nx.data(gt=0)
        y: int = nx.data(gt=0)

    return PositivePoint


@pytest.fixture(scope="module")
def SimpleFix():
    """A full record schema with key and timestamp."""

    class SimpleFix(Sample):
        mobile: int = nx.key()
        time: datetime = nx.timestamp()
        x: float = nx.data()
        y: float = nx.data()

    return SimpleFix


@pytest.fixture(scope="module")
def Fix(PositiveFloat):
    """Fixture intended to test embedding of compound fields."""

    class Fix(Sample):
        mobile: int = nx.key()
        time: datetime = nx.timestamp(tz="utc")
        x: float = nx.data()
        y: float = nx.data()
        z: PositiveFloat = nx.data()

    return Fix


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def DefaultRecord(WeirdInteger, PositiveFloat):
    """Fixture that tests the use of default alongside compound fields."""

    class DefaultRecord(Record):
        key: WeirdInteger = nx.key()
        x: PositiveFloat = nx.data(1.0)
        y: PositiveFloat = nx.data(PositiveFloat(1.0))

    return DefaultRecord


@pytest.fixture(scope="module")
def TempJournal():
    class TempJournal(Journal):
        """A daily temperature summary."""

        machine_id: int = nx.key()
        day: pd.Period = nx.period(freq="D")
        mean_temp: float = nx.data()

    return TempJournal


@pytest.fixture(scope="module")
def Trip():
    class Trip(Session):
        """A daily temperature summary."""

        mobile: int = nx.key()
        start_time: pd.Timestamp = nx.start_time(tz="UTC")
        end_time: pd.Timestamp = nx.end_time(tz="UTC")

    return Trip


def test_data_object_base(WeirdInteger, PositiveFloat):
    assert WeirdInteger.dataspec.pytype is int
    assert list(WeirdInteger.options) == ["parse", "conform", "validate", "integrate"]
    assert WeirdInteger.validate
    subtype = Data.subtype(dataspec=WeirdInteger.dataspec)
    assert subtype is WeirdInteger
    assert WeirdInteger.__kwargs__ == {
        "dataspec",
        "parse",
        "conform",
        "validate",
        "integrate",
    }
    assert issubclass(Data, nxdata)
    assert issubclass(Data.Base, nxdata)
    assert issubclass(WeirdInteger, nxdata)
    assert issubclass(PositiveFloat, nxdata)
    assert (
        repr(PositiveFloat.typespec)
        == "<typespec:Data|dataspec=DataSpec(pytype=<class 'float'>, name='PositiveFloat', nullable=False)>"
    )


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


def test_model(Point, PositivePoint, XPoint):
    assert issubclass(Point, nxmodel)
    assert list(Point.dataschema.fields) == ["x", "y"]
    assert Point.__kwargs__ == {
        "dataschema",
        "strict_schema",
        "parse",
        "conform",
        "validate",
        "integrate",
        "x",
        "y",
    }
    data = {"x": 0, "y": 0}
    point = Point(data)
    assert dict(point.data) == data
    assert Point(x=0, y=0) == point
    assert Point(x="0", y="0") == point
    assert point.x == Data[Point.x](0) == Integer(0) == Data[int](0)
    with pytest.raises(ValueError):
        PositivePoint(x=-1, y=1)
    assert PositivePoint(x=-1, y=1, validate=False).x.data == -1
    xdata = {"x": 0, "y": 0, "z": 0}
    point = Point(xdata)
    assert dict(point.data) == data
    xpoint = XPoint(xdata)
    assert dict(xpoint.data) == xdata


def test_column(PositiveFloat, WeirdInteger):
    s0 = [1, 2, 3]
    s1 = [0, 1, 2]
    s2 = [None, 1]
    s3 = ["a", "b", "c"]
    s4 = [1, 2, 1e10]
    PositiveFloatSeries = Column.subtype(datatype=PositiveFloat)
    S0 = PositiveFloatSeries(s0)
    pd.Series(s0).equals(S0.data)
    with pytest.raises(ValueError):
        PositiveFloatSeries(s1)
    # This raises a ValueError and not a TypeError because pandas preprocesses
    # s2 into [NaN, 1.0] and float accepts NaN as an argument. Not sure whether
    # this is the desired behavior or not, so leaving this open-ended for now.
    with pytest.raises(ValueError):
        PositiveFloatSeries(s2)
    # This on the other hand unambiguously raises a TypeError.
    with pytest.raises(TypeError):
        PositiveFloatSeries(s3)
    with pytest.raises(ValueError):
        PositiveFloatSeries(s4)
    WeirdIntegerSeries = Column[WeirdInteger]
    S0 = WeirdIntegerSeries(s0)
    assert list(S0.data) == s0
    S3 = WeirdIntegerSeries(data=s3)
    assert all(isinstance(i, int) for i in S3.data)


def test_table(Point, PositivePoint, XPoint):
    PointTable = Table[Point]
    PositivePointTable = Table[PositivePoint]
    XPointTable = Table[XPoint]
    data = {"x": [0, 0, 0], "y": [0, 1, 2]}
    point_table = PointTable(data)
    assert pd.DataFrame(data).equals(point_table.data)
    with pytest.raises(ValueError):
        PositivePointTable(data)
    xdata = {"x": [0, 0, 0], "y": [0, 1, 2], "z": [0, 0, 0]}
    xpoint_table = XPointTable(xdata)
    assert pd.DataFrame(xdata).equals(xpoint_table.data)
    mdata = {"x": [0, 0, 0]}
    with pytest.raises(TypeError):
        PointTable(mdata, parse=False)
    tdata = {"x": ["0", "0", "0"], "y": [0, 1, 2]}
    tpoint_table = PointTable(tdata)
    # Here the data gets parsed into integer
    assert pd.DataFrame(data).equals(tpoint_table.data)
    # In this case parsing is skipped, but conforming does the job column-wise
    no_parse_tpoint_table = PointTable(tdata, parse=False)
    assert pd.DataFrame(data).equals(no_parse_tpoint_table.data)
    # Non-conformable data
    bdata = {"x": ["a", "0", "0"], "y": [0, 1, 2]}
    with pytest.raises(TypeError):
        PointTable(bdata, parse=False)
    # This tests attributes of the table
    assert point_table.x.data.equals(point_table.data.x)
    assert list(point_table.x.data) == [0, 0, 0]
    assert (
        point_table.x
        == Column[Point.x]([0, 0, 0])
        == Column[Integer]([0, 0, 0])
        == Column[int]([0, 0, 0])
    )


def test_record_index(SimpleFix):
    assert dict(SimpleFix.dataschema.index) == {
        "nx_key": ("mobile",),
        "nx_timestamp": ("time",),
    }
    assert SimpleFix.dataschema.is_record_schema
    assert issubclass(SimpleFix, Record)
    assert issubclass(SimpleFix, Sample)
    fix_data = {
        "mobile": 0,
        "time": datetime.strptime("2022-02-08 16:31", "%Y-%m-%d %H:%M"),
        "x": 5,
        "y": 3,
    }
    fix = SimpleFix(**fix_data)
    assert fix.x == Data[SimpleFix.x](5)
    assert fix.x.data == 5
    with pytest.raises(TypeError):

        class BadFix(Sample):
            mobile: int = nx.key()
            time: int = nx.timestamp()


def test_table_index(SimpleFix):
    SimpleFixTable = Table[SimpleFix]
    assert SimpleFixTable.dataschema is SimpleFix.dataschema
    data = {
        "mobile": [0, 0],
        "time": [
            datetime.strptime("2022-02-08 16:31", "%Y-%m-%d %H:%M"),
            datetime.strptime("2022-02-08 16:32", "%Y-%m-%d %H:%M"),
        ],
        "x": [5, 7],
        "y": [3, 1],
    }
    table = SimpleFixTable(data)
    assert pd.Series([5.0, 7.0], name="x").equals(table.data.reset_index().x)
    assert isinstance(table.data.index, pd.MultiIndex)
    assert table.data.index.names == table.x.data.index.names == ["mobile", "time"]
    assert table.mobile.data.equals(table.data.reset_index().mobile)
    assert table.mobile == Column[SimpleFix.mobile]([0, 0])
    assert table.mobile.data.equals(
        pd.Series(data["mobile"], name="mobile").astype("category")
    )
    assert table.time.data.equals(pd.Series(data["time"], name="time"))
    scoped_table = SimpleFixTable(data, datascope={"keys": [0, 1]})
    assert isinstance(scoped_table.datascope, DataScope)
    assert list(scoped_table.data.reset_index().mobile.dtype.categories) == [0, 1]


def test_compound_model(Fix, PositiveFloat):
    """Tests Fix, one of which fields is a Data type."""
    fix_data = {
        "mobile": 0,
        "time": datetime.strptime("2022-02-08 16:31", "%Y-%m-%d %H:%M"),
        "x": 5,
        "y": 3,
        "z": 1,
    }
    fix = Fix(**fix_data)
    assert fix.x.data == 5
    assert fix.z == PositiveFloat(1)
    assert fix.data.z == fix.z.data == 1
    assert fix.time.data.tzinfo == pytz.utc
    bad_data = fix_data | dict(z=-1)
    with pytest.raises(ValueError):
        Fix(**bad_data)
    bad_data = fix_data | dict(z=1e10)
    with pytest.raises(ValueError):
        Fix(**bad_data)


def test_compound_validation(WeirdInteger, WeirdRecord, PositiveFloat):
    """Tests compound validation scenarios."""
    input_data = dict(key=0, i=10, j=10, inverter=-1)
    output_data = input_data | dict(inverter=1)
    record = WeirdRecord(**input_data)
    assert record.key == WeirdInteger(0)
    assert dict(record.data) == output_data
    alt_key = input_data | dict(key=12)
    assert WeirdRecord(**alt_key).key.data == 12
    string_data = input_data | dict(i="i")
    assert WeirdRecord(**string_data).i == WeirdInteger("i")
    partial_data = input_data | dict(inverter=None)
    assert WeirdRecord(**partial_data).key.data == 0
    bad_key = input_data | dict(key=2)
    with pytest.raises(ValueError):
        WeirdRecord(**bad_key)
    WeirdRecord(**bad_key, validate=False).key == WeirdInteger(2)
    other_bad_key = input_data | dict(key=21)
    with pytest.raises(ValueError):
        WeirdRecord(**other_bad_key)
    bad_inverter = input_data | dict(inverter=1)
    with pytest.raises(ValueError):
        WeirdRecord(**bad_inverter)
    bad_i = input_data | dict(i=5)
    with pytest.raises(ValueError):
        WeirdRecord(**bad_i)
    bad_j = input_data | dict(i=5)
    with pytest.raises(ValueError):
        WeirdRecord(**bad_j)
    # This raises because the field specifies a validation value
    # that contradicts PositiveFloat's built-in validation
    with pytest.raises(ValueError):

        class ContradictoryRecord(Record):
            x: PositiveFloat = nx.data(le=1e10)


def test_compound_default(DefaultRecord, PositiveFloat):
    record = DefaultRecord(key=0)
    data = dict(key=0, x=1.0, y=1.0)
    assert dict(record.data) == data
    assert record.y == PositiveFloat(1.0)


def test_period(TempJournal):
    data = {"machine_id": 0, "day": "2022-03-24", "mean_temp": 75.5}
    journal = TempJournal(**data)
    assert journal.day.data == journal.data.day == pd.Period("2022-03-24", "D")
    typed_data = data | {"day": pd.Period("2022-03-24", "D")}
    assert TempJournal(**typed_data) == journal
    assert TempJournal(journal) == journal
    assert TempJournal(**journal.data) == journal
    objected_data = data | {"day": journal.day}
    assert TempJournal(**objected_data) == journal
    TempJournalTable = Table[TempJournal]
    table = TempJournalTable(pd.DataFrame.from_records([data]))
    assert list(table.data.index.dtypes.apply(str)) == ["category", "period[D]"]


def test_session(Trip):
    data_0 = {
        "mobile": 0,
        "start_time": "2022-03-24 18:30",
        "end_time": "2022-03-24 23:00",
    }
    data_1 = {
        "mobile": 1,
        "start_time": "2022-03-25 10:30",
        "end_time": "2022-03-25 23:00",
    }
    trip_0 = Trip(data_0)
    trip_1 = Trip(**data_1)
    assert trip_0.timespan.length == pd.Timedelta("4:30:00")
    trips = pd.DataFrame.from_records([data_0, data_1])
    table = Table[Trip](trips)
    assert list(table.data.index.dtypes.apply(str)) == [
        "category",
        "interval[datetime64[ns, UTC], left]",
    ]
    assert table.start_time.data.equals(
        pd.Series(["2022-03-24 18:30", "2022-03-25 10:30"], name="start_time").astype(
            "datetime64[ns, UTC]"
        )
    )
    # There is inequality because the left-hand operand is indexed
    assert not table.data.start_time.equals(table.start_time.data)
