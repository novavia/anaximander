from datetime import datetime

import pandas as pd
import pytest
import pytz

import anaximander as nx
from anaximander.descriptors.datatypes import nxmodel

from anaximander.dataobjects import Data, Integer, Table, Record, Sample


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
    assert (table.timeblocks == pd.Series(pd.Period("2022-03-24", "D"))).all()
    assert TempJournalTable(
        pd.DataFrame.from_records([data]), datascope={"time": ["2022-03", "2022-04"]}
    )
    with pytest.raises(ValueError):
        TempJournalTable(
            pd.DataFrame.from_records([data]),
            datascope={"time": ["2022-02", "2022-03"]},
        )


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
    assert table.start_times.equals(pd.DatetimeIndex(table.data.start_time))
    assert table.end_times.equals(pd.DatetimeIndex(table.data.end_time))
    assert table.timespans[0] == trip_0.timespan
    assert Table[Trip](trips, datascope={"time": ["2022-03", "2022-04"]})
    with pytest.raises(ValueError):
        Table[Trip](trips, datascope={"time": ["2022-02", "2022-03"]})
