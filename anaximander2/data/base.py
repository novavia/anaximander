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
from pandas.api.types import CategoricalDtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype

from ..utilities import functions as fun
from ..structures import Structure
from . import fields


__all__ = []

# =============================================================================
# Base DataObject
# =============================================================================


class DataObject(Structure):
    """Base class for all data objects."""

    def __init__(self, data, metadata=None, cast=True):
        if cast:
            self._data = self.cast(data)
        else:
            self._data = data
        self._metadata = fun.get(metadata, {})

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata.copy()

    @abc.abstractclassmethod
    def cast(cls, data):
        return data


# =============================================================================
# Pandas mapping
# =============================================================================


cat_match = fields.TypeMapper(lambda f: CategoricalDtype(f.categories,
                                                         f.ordered))
dt_match = fields.TypeMapper(lambda f: DatetimeTZDtype(tz=f.tz)
                             if f.tz else np.dtype('datetime64[ns]'))

_pandas_field_mapping = {fields.NxField: np.dtype('object'),
                         fields.Numeric: np.dtype('float'),
                         fields.Integer: np.dtype('int'),
                         fields.Float: np.dtype('float'),
                         fields.String: np.dtype('object'),
                         fields.Text: np.dtype('object'),
                         fields.Categorical: cat_match,
                         fields.State: cat_match,
                         fields.EventType: cat_match,
                         fields.Date: dt_match,
                         fields.Time: dt_match,
                         fields.DateTime: dt_match,
                         fields.Timestamp: np.dtype('int'),
                         fields.ObjectID: np.dtype('int')}

pd_type_map = fields.TypeMap(_pandas_field_mapping)
