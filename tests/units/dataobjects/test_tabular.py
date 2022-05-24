from datetime import datetime
import pandas as pd
import pytest

from anaximander.descriptors.scope import DataScope

from anaximander.dataobjects import Integer, Column, Table


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

    # Test slices
    assert point_table._row_from_iloc(0) == Point(x=0, y=0)
    assert point_table._slice_from_iloc(0, 3) == point_table
    assert point_table._slice_from_iloc(0, 1) == PointTable(dict(x=[0], y=[0]))


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
    mobile_column = Column[SimpleFix.mobile]([0, 0])
    assert table.mobile == Column[SimpleFix.mobile]([0, 0])
    assert table.mobile.data.equals(
        pd.Series(data["mobile"], name="mobile").astype("category")
    )
    assert table.time.data.equals(pd.Series(data["time"], name="time"))
    assert (table.timestamps == pd.Series(data["time"])).all()
    key_scoped_table = SimpleFixTable(data, datascope={"keys": [0, 1]})
    assert isinstance(key_scoped_table.datascope, DataScope)
    assert list(key_scoped_table.data.reset_index().mobile.dtype.categories) == [0, 1]
    with pytest.raises(ValueError):
        SimpleFixTable(data, datascope={"keys": [1]})
    time_scoped_table = SimpleFixTable(data, datascope={"time": ["2022-02", "2022-03"]})
    assert isinstance(time_scoped_table.datascope["time"], pd.Interval)
    with pytest.raises(ValueError):
        SimpleFixTable(data, datascope={"time": ["2022-01", "2022-02"]})

    d0 = {k: v[0] for k, v in data.items()}
    assert table._row_from_iloc(0) == SimpleFix(**d0)
    assert table._slice_from_iloc(0, 3) == table
    d01 = {k: v[0:1] for k, v in data.items()}
    assert table._slice_from_iloc(0, 1) == SimpleFixTable(d01)
    assert next(table.records()) == SimpleFix(**d0)
