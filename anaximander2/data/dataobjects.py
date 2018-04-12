#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the data object base type.

Data objects are immutable containers of columnar data in various topologies.
At the highest level, data objects are declined in four archetypes, namely
data frames (i.e. matrix), data series (single column), data records (single
row) and data observations (scalar).

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype

from ..utilities import functions as fun, xprops
from ..structures import NxStructure
from ..meta import NxObject, archetype, TypeParameter
from . import columns
from . import schema as sch


__all__ = []

# =============================================================================
# Base DataObject
# =============================================================================


class ConformityError(Exception):
    """Raised if data supplied to a DataObject doesn't conform."""
    pass


class DataObject(NxStructure):
    """Base class for all data objects."""
    __schema__ = None  # placeholder for specialized type parameter

    def __init__(self, data, *, id_range=None, sq_range=None, cast=True,
                 schema=None, **metadata):
        self.schema = fun.get(schema, self.__schema__())
        if cast:
            self._data = self.cast(data)
        else:
            self._data = data
        if self.identifier in metadata:
            self._id_range = metadata.pop(self.identifier)
        else:
            self._id_range = id_range
        if self.sequencer in metadata:
            self._sq_range = metadata.pop(self.sequencer)
        else:
            self._sq_range = sq_range
        self._metadata = metadata

    @xprops.cachedproperty
    def indexing(self):
        ixtype = self.__schema__.__index__
        if isinstance(ixtype, sch.NominalSchemaIndexType):
            return 'nominal'
        elif isinstance(ixtype, sch.SequentialSchemaIndexType):
            return 'sequential'
        elif isinstance(ixtype, sch.DualSchemaIndexType):
            return 'dual'

    @property
    def identifier(self):
        return self.schema.index.identifier

    @property
    def sequencer(self):
        return self.schema.index.sequencer

    @property
    def data(self):
        return self._data.copy()

    @property
    def metadata(self):
        rval = self._metadata.copy()
        rval[self.identifier] = self._id_range
        rval[self.sequencer] = self._sq_range
        return rval

    @property
    def id_range(self):
        return self._id_range

    @property
    def sq_range(self):
        return self._sq_range

    @abc.abstractclassmethod
    def cast(cls, data):
        return data


class IndexedDataObject:
    """Base mix-in class for NxDataFrame and NxDataColumn."""

    @property
    def index(self):
        """Returns the object's index."""
        return self._data.index.copy()

    @abc.abstractproperty
    def idx(self):
        """Returns a series of indexing tuples."""
        return pd.Series([()], index=self.index)

    @property
    def key(self):
        """Returns a series of row keys, indexed by self's index."""
        return self.idx.apply(self.schema.rowkey).rename('key')

    @abc.abstractmethod
    def __getitem__(self, key):
        """Slices the data per index properties."""
        return NotImplemented


class NominallyIndexedDataObject(IndexedDataObject):
    """Mix-in class for nominally-indexed objects."""

    @property
    def idx(self):
        if self.indexing == 'nominal':
            idx_ = list(zip(self.index))
        elif self.indexing == 'dual':
            idx_ = list(zip(self.index, [self.sq_range]))
        return pd.DataFrame({'idx': idx_}, index=self.index).idx


class SequentiallyIndexedDataObject(IndexedDataObject):
    """Mix-in class for sequentialy-indexed objects."""

    @property
    def idx(self):
        if self.indexing == 'sequential':
            idx_ = list(zip(self.index))
        elif self.indexing == 'dual':
            idx_ = list(zip([self.id_range], self.index))
        return pd.DataFrame({'idx': idx_}, index=self.index).idx


class DoublyIndexedDataObject(IndexedDataObject):
    """Mix-in class for doubly-indexed objects."""

    @property
    def idx(self):
        return pd.DataFrame({'idx': list(self.index)}, index=self.index).idx


class SingleRowDataObject(DataObject):
    """Base class for NxRecord and NxField."""

    @property
    def idx(self):
        """Returns a tuple, per the object's schema index."""
        if self.indexing == 'nominal':
            return (self.id_range,)
        elif self.indexing == 'sequential':
            return (self.sq_range,)
        elif self.indexing == 'dual':
            return (self.id_range, self.sq_range)

    @property
    def key(self):
        """Returns the row key corresponding to self's index."""
        return self.schema.rowkey(self.idx)

# =============================================================================
# Pandas mapping
# =============================================================================


pd_cat_match = columns.TypeMapper(lambda f: CategoricalDtype(f.categories,
                                                             f.ordered))
pd_dt_match = columns.TypeMapper(lambda f: DatetimeTZDtype(tz=f.tz)
                                 if f.tz else np.dtype('datetime64[ns]'))

_pandas_mapping = {columns.NxColumn: np.dtype('object'),
                   columns.Numeric: np.dtype('float'),
                   columns.Integer: np.dtype('int'),
                   columns.Float: np.dtype('float'),
                   columns.String: np.dtype('object'),
                   columns.Text: np.dtype('object'),
                   columns.Categorical: pd_cat_match,
                   columns.State: pd_cat_match,
                   columns.EventType: pd_cat_match,
                   columns.Date: pd_dt_match,
                   columns.Time: pd_dt_match,
                   columns.DateTime: pd_dt_match,
                   columns.Timestamp: np.dtype('int'),
                   columns.ObjectID: np.dtype('int')}

pd_type_map = columns.TypeMap(_pandas_mapping)


# =============================================================================
# Fields mapping
# =============================================================================


@archetype
class ValidatedString(NxObject):
    enum: tuple = TypeParameter()

    def __new__(cls, object=''):
        if object in cls.enum:
            return str(object)
        raise ValueError


fd_cat_match = columns.TypeMapper(lambda f: ValidatedString[f.categories])


_fields_mapping = {columns.NxColumn: np.dtype('object').type,
                   columns.Numeric: np.dtype('float').type,
                   columns.Integer: np.dtype('int').type,
                   columns.Float: np.dtype('float').type,
                   columns.String: np.dtype('object').type,
                   columns.Text: np.dtype('object').type,
                   columns.Categorical: fd_cat_match,
                   columns.State: fd_cat_match,
                   columns.EventType: fd_cat_match,
                   columns.Date: None,
                   columns.Time: None,
                   columns.DateTime: None,
                   columns.Timestamp: np.dtype('int'),
                   columns.ObjectID: np.dtype('int')}

fd_type_map = columns.TypeMap(_fields_mapping)
