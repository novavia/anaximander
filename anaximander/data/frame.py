#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a base Frame class.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from collections import OrderedDict

import numpy as np
import pandas as pd

from ..meta.metadescriptors import MetaCharacter
from ..meta.nxtype import prototype
from .exceptions import DataError
from .base import IndexedDataObject, DataSlicingError
from .schema import Schema
from .fields import Field, Raw, Nested, Dict, List, String, UUID, \
    Number, Integer, Decimal, Boolean, FormattedString, Float, DateTime, \
    LocalDateTime, Time, Date, TimeDelta, Url, URL, Email, Method, Function, \
    Str, Bool, Int, Constant, Scalar
from .record import NxRecord
from .series import NxSeries

# =============================================================================
# Field mapping from Schemas to pandas / numpy
# =============================================================================


# TODO: add Scalar
_field_map = {Field: np.dtype('object'),
              Raw: np.dtype('object'),
              Nested: np.dtype('object'),
              Dict: NotImplemented,
              List: NotImplemented,
              String: np.dtype('object'),
              UUID: np.dtype('object'),
              Number: np.dtype('float'),
              Integer: np.dtype('int'),
              Decimal: np.dtype('float'),
              Boolean: np.dtype('bool'),
              FormattedString: np.dtype('object'),
              Float: np.dtype('float'),
              DateTime: np.dtype('datetime64[ns]'),
              LocalDateTime: np.dtype('datetime64[ns]'),
              Time: np.dtype('datetime64[ns]'),
              Date: np.dtype('datetime64[ns]'),
              TimeDelta: np.dtype('timedelta64[ns]'),
              Url: np.dtype('object'),
              URL: np.dtype('object'),
              Email: np.dtype('object'),
              Method: NotImplemented,
              Function: NotImplemented,
              Str: np.dtype('object'),
              Bool: NotImplemented,
              Int: NotImplemented,
              Constant: NotImplemented,
              }


def dtype(field):
    """Returns a BiqQuery field type from a Schema field or NotImpemented."""
    ftype = type(field)
    if issubclass(ftype, Scalar):
        return field.datatype.dtype
    for ft in ftype.__mro__:
        try:
            return _field_map[ft]
        except KeyError:
            pass
    raise TypeError("Non-field type {0} passed to dtype.".format(field))


def dtypes(schema):
    """Turns a schema class or instance into an OrderedDict of dtypes."""
    return OrderedDict([(k, dtype(v)) for k, v in schema.fields.items()])


class FrameError(DataError):
    """Specialized exception for Frames."""
    pass


class ConformityError(FrameError):
    """Raised at Frame construction if supplied data doesn't conform."""
    pass

# =============================================================================
# Frame base class
# =============================================================================


class _FrameIndexProxy(object):
    """Wraps pandas indexer object to return NxData objects."""

    def __init__(self, nxdata, pdidx):
        """Instantiated with a NxFrame and a pandas accessor.

        A pandas accessor is any pandas object that implement __getitem__
        on the underlying series with which the instance was created.
        """
        self._nxdata = nxdata
        self._pdidx = pdidx

    def __getitem__(self, key):
        data = self._pdidx.__getitem__(key)
        context = self._nxdata.context
        if isinstance(data, pd.DataFrame):
            try:
                return type(self._nxdata)(data, context=context)
            except ConformityError:
                msg = "Slice cannot be cast into a DataObject."
                raise DataSlicingError(msg)
        if isinstance(data, pd.Series):
            # If index is subset of the frame's columns, then data is a record
            if set(data.index).issubset(set(self._nxdata.schema.fields)):
                rtype = NxRecord[self._nxdata.schema]
                return rtype(data, context=context)
            else:
                stype = NxSeries[self._nxdata.schema]
                return stype(data, context=context)
        # Otherwise the function returns a scalar but since we don't
        # know where it's coming from (i.e. NxData or not) the function
        # raises an exception.
        # This is clearly a flaw but to fix it would require delving deep
        # into pandsas indexing and intercepting all the scenarios by
        # which a scalar is returned through direct slicing of a DataFrame.
        # Two alternative possibilities exist:
        # * Directly slice self.data to get a numpy scalar
        # * Do a slice of a slice, i.e. either slice a column or slice
        # a row, which will return an NxData object or a numpy object
        # as appropriate.
        else:
            raise DataSlicingError(msg)


@prototype
class NxFrame(IndexedDataObject):

    schema = MetaCharacter(validate=lambda s: issubclass(s, Schema))

    def __init__(self, data, context=None, validate=False):
        """Data can be any admissible data argument to a dataframe.

        params:
            data: a DataFrame of data argument to a DataFrame.
            context: Optional context to populate DataObject's context
                weak property.
            validate: if True, the entire data gets validated against the
                class' schema. Must be used intentionally as there is a
                performance penalty.

        If validate is False, the method nonetheless performs a schema-level
        validation to verify conformity between the DataFrame's column names
        and types and the class' Schema (see cast method).
        """
        if isinstance(data, NxFrame):
            context = context or data.context
            data = data.data
        self._data = self.cast(data)
        if validate:
            self.validate()

    @property
    def data(self):
        return self._data.copy()

    @classmethod
    def cast(cls, data):
        """Casts supplied dataframe-like object to the class' schema.

        data must be a valid input to pandas.DataFrame.
        The method performs the following functions:
        * raises PandasError if data cannot be cast to a DataFrame
        * raises ConformityError if the data misses required columns;
        on the other hand, no error is raised if other columns are missing,
        and extra columns are simply removed.
        * recasts columns to the dtype specified in the schema if necessary;
        * reorders columns to match the schema if necessary.
        """
        df = pd.DataFrame(data)
        df_dtypes = OrderedDict(df.dtypes)
        columns = []
        conversions = {}
        sc_fields = cls.schema.fields.items()
        sc_dtypes = dtypes(cls.schema).items()
        for (col, field), (_, dtype) in zip(sc_fields, sc_dtypes):
            try:
                col_type = df_dtypes[col]
            except KeyError:
                if field.required:
                    msg = "Data is missing required field."
                    raise ConformityError(msg)
            else:
                columns.append(col)
                if col_type != dtype:
                    conversions[col] = dtype
        dataframe = df[columns]
        if conversions:
            try:
                dataframe = dataframe.astype(conversions)
            except ValueError:
                msg = "Data cannot be cast to required types."
                raise ConformityError(msg)
        return dataframe

    def validate(self):
        """Validates all records in a frame against the schema."""
        records = self.data.astype(str).to_dict(orient='records')
        schema = self.schema(many=True)
        schema.validate(records)
