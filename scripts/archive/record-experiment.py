from datetime import date
from pprint import pprint
from typing import Iterable, Optional

from marshmallow import Schema, fields, ValidationError, missing
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

PYTYPES = {
    fields.Str: str,
    fields.Date: date,
    fields.Int: int,
    fields.List: list,
    fields.Nested: object,
}


class ArtistSchema(Schema):
    name = fields.Str()


class AlbumSongSchema(Schema):
    album = fields.Nested(lambda: AlbumSchema(only=("title",)))
    title = fields.Str()
    track_number = fields.Int()


class AlbumSchema(Schema):
    title = fields.Str()
    release_date = fields.Date()
    artist = fields.Nested(ArtistSchema())
    songs = fields.List(fields.Nested(AlbumSongSchema(exclude=("album",))))


ECLIPSE_DATA = {
    "title": "Eclipse",
    "release_date": "2014-11-27",
    "artist": {"name": "Jr Dreads"},
    "songs": [
        {"title": "Downstream", "track_number": 1},
        {"title": "The Grass", "track_number": 2},
        {"title": "Sweetie", "track_number": 3},
        {"title": "Give Me Love", "track_number": 4},
        {"title": "Eclipse", "track_number": 5},
    ],
}

album_schema = AlbumSchema()

ECLIPSE = album_schema.load(ECLIPSE_DATA)
assert ECLIPSE["release_date"] == date(2014, 11, 27)


class DataObject:
    pass

    def __call__(self):
        return self.data


class Cell(DataObject):
    def __init__(
        self, data, *, field: Optional[fields.Field], cast=True, validate=True
    ):
        self.field = field
        if cast:
            value = self.cast(data)
        self._data = self.cast(data)
        if validate:
            self.validate()

    @property
    def data(self):
        return self._data

    def cast(self, data):
        if self.field is None:
            return data
        if data is None and not self.field.allow_none:
            raise ValidationError()
        dtype = PYTYPES.get(type(self.field), object)
        if not isinstance(data, dtype):
            try:
                data = dtype(data)
            except (TypeError, ValueError):
                raise ValidationError()
        return data

    def validate(self):
        return True

    def __repr__(self):
        if self.field is None:
            return "<Cell>"
        else:
            return f"<Cell[{type(self.field).__name__}]>"

    def __str__(self):
        return str(self._data)


class Record(DataObject):
    def __init__(
        self, data, *, schema: Optional[Schema] = None, cast=True, validate=True
    ):
        self.schema = schema
        if cast:
            data = self.cast(data)
        self._data = data
        if validate:
            self.validate()

    def cast(self, data):
        if self.schema is None:
            return data
        result = {}
        for k, v in self.schema.fields.items():
            if k not in data:
                if v.required:
                    raise ValidationError()
                elif v.missing is not missing:
                    result[k] = v.missing
                else:
                    continue
            value = data[k]
            if value is None and not v.allow_none:
                raise ValidationError()
            dtype = PYTYPES.get(type(v), object)
            if not isinstance(value, dtype):
                try:
                    value = dtype(value)
                except (TypeError, ValueError):
                    raise ValidationError()
            result[k] = value
        return result

    def validate(self):
        return True

    @property
    def data(self):
        return self._data.copy()

    def __getattr__(self, attr):
        try:
            field = self.schema.fields[attr]
            value = self._data[attr]
        except KeyError:
            raise AttributeError
        else:
            return Cell(value, field=field, cast=False, validate=False)

    def __repr__(self):
        if self.schema is None:
            return "<Record>"
        else:
            return f"<Record[{type(self.schema).__name__}]>"

    def __str__(self):
        base_string = str(pd.Series(self._data))
        return base_string[: base_string.rfind("\n")]


JRDREADS = Record({"name": "Jr Dreads"}, schema=ArtistSchema())
