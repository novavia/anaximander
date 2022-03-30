from datetime import date
from pprint import pprint
from typing import Iterable, Optional

from marshmallow import Schema, fields, ValidationError, missing
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

DTYPES = {
    fields.Str: pd.StringDtype,
    fields.Int: np.dtype("int"),
    fields.Nested: np.dtype("object"),
    fields.List: np.dtype("object"),
    fields.Bool: np.dtype("bool"),
    fields.Date: np.dtype("datetime64[ns]"),
}


def dtypes(schema: Schema):
    return {
        k: DTYPES.get(type(v), np.dtype("object")) for k, v in schema.fields.items()
    }


class ArtistSchema(Schema):
    name = fields.Str()


class AlbumSongSchema(Schema):
    album = fields.Nested(lambda: AlbumSchema(only=("title",)))
    title = fields.Str()
    track_number = fields.Int()
    explicit = fields.Bool()


class AlbumSchema(Schema):
    title = fields.Str()
    release_date = fields.Date()
    artist = fields.Nested(ArtistSchema())
    songs = fields.List(fields.Nested(AlbumSongSchema(exclude=("album",))))


jrdreads = {"name": "Jr Dreads"}

downstream = {"album": {"title": "Eclipse"}, "title": "Downstream", "track_number": 1}
the_grass = {"album": {"title": "Eclipse"}, "title": "The Grass", "track_number": 2}
sweetie = {"album": {"title": "Eclipse"}, "title": "Sweetie", "track_number": 3}
give_me_love = {
    "album": {"title": "Eclipse"},
    "title": "Give Me Love",
    "track_number": 4,
}
eclipse = {"album": {"title": "Eclipse"}, "title": "Eclipse", "track_number": 5}

eclipse_album = {
    "title": "Eclipse",
    "release_date": date(2014, 11, 27),
    "artist": jrdreads,
    "songs": [downstream, the_grass, sweetie, give_me_love, eclipse],
}

eclipse_df = pd.DataFrame(
    {
        "album": "Eclipse",
        "title": [s["title"] for s in eclipse_album["songs"]],
        "track_number": [s["track_number"] for s in eclipse_album["songs"]],
        "explicit": False,
    }
)


class Table:
    def __init__(
        self, data: Optional[pd.DataFrame] = None, *, schema: Optional[Schema] = None
    ):
        conformed = self._conform(data, schema)
        self._data = conformed

    @classmethod
    def _empty_frame(cls, schema: Schema):
        return pd.DataFrame(columns=schema.fields)

    # TODO: the main conceptual problem here is what happens if heterogeneous
    # data is loaded into an input data frame such that pandas automatically
    # assigns nan values to records that were missing fields. This creates a
    # hole because these fields would either have been invalidated, or their
    # missing values filled in if they had been deserialized by marshmallows.
    # This is at least one discrepancy in semantics that is not addressed
    # in this implementation.
    # Further, one seemingly common need that is not addressed out of the box
    # by Marshmallow is filling None values -presumably with what is set
    # in the missing parameter.
    @classmethod
    def _conform(cls, data: Optional[pd.DataFrame], schema: Optional[Schema]):
        if data in (None, [], (), {}):
            data = cls._empty_frame(schema)
        assert isinstance(data, pd.DataFrame)
        data_types = data.dtypes
        if schema is None:
            return data
        columns = []
        for k, v in schema.fields.items():
            if k not in data:
                if v.required:
                    raise ValidationError()
                elif v.missing is not missing:
                    data[k] = v.missing
                else:
                    continue
            columns.append(k)
            if not v.allow_none:
                if data[k].isna().any():
                    raise ValidationError()
            dtype = DTYPES.get(type(v), np.dtype("object"))
            if not data_types[k] is dtype:
                data[k] = data[k].astype(dtype)
        conformed = data[columns]
        return conformed
