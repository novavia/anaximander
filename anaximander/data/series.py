#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Series module, which defines the NxSeries archetype.

This version only supports univariate data types. Support for multi-variate
types based on compound numpy dtypes is planned for future release.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from collections.abc import Sequence
from functools import partial

import pandas as pd

from anaximander.utilities.functions import spformat
from anaximander.meta.nxtype import prototype
from anaximander.meta.metadescriptors import MetaCharacter, typeproperty
from .object import DataObject
from .data import NxData

# =============================================================================
# NxSeries prototype
# =============================================================================


class _Accessor(object):
    """Wraps pandas accessors to return NxData objects."""

    def __init__(self, series, pdaccessor):
        """Instantiated with a NxSeries and a pandas accessor.

        A pandas accessor is any pandas object that implement __getitem__
        on the underlying series with which the instance was created.
        """
        self._series = series
        self._pdaccessor = pdaccessor

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self._series)(self._pdaccessor.__getitem__(key),
                                      **self._series.metadata)
        return self._series.datatype(self._pdaccessor.__getitem__(key),
                                     **self._series.metadata)


@prototype
class NxSeries(DataObject, overtype=True, traits=(Sequence,)):
    """A read-only indexed sequence of NxData.

    NxSeries exposes a pandas Series, whose elements are of type
    self.datatype.dtype, i.e. the dtype of the NxDataType that is the
    metacharacter value of concrete NxSeries subtypes.
    """
    datatype = MetaCharacter(validate=lambda s: issubclass(s, NxData))

    @typeproperty
    def dtype(cls):
        return cls.datatype.dtype

    @typeproperty
    def quantity(cls):
        return cls.datatype.quantity

    @typeproperty
    def unit(cls):
        return cls.datatype.unit

    def __init__(self, data, index=None, **metadata):
        # XXX: Eventually, add the possibility that what is passed to the
        # constructor is a Sequence of NxData. If this is done right, such
        # a collection (based on Sequence being a prototype) should carry
        # a data collectiveproperty, which results in the same data.data
        # assignment covering that use case.
        if isinstance(data, NxSeries):
            data = data.data
        self._data = pd.Series(data, dtype=self.datatype.dtype, copy=True)
        self._metadata = metadata

    @property
    def data(self):
        return self._data.copy()

    @property
    def metadata(self):
        return self._metadata

    def __getitem__(self, key):
        """Similar to pandas.Series, label-based, but returns NxData object."""
        if isinstance(key, slice):
            return type(self)(self._data.__getitem__(key), **self.metadata)
        return self.datatype(self._data.__getitem__(key), **self.metadata)

    def __len__(self):
        return self._data.size

    def __repr__(self):
        left_str_base = '<NxSeries[{dt}]({content}'
        left_str = left_str_base.format(dt=self.datatype.__name__,
                                        content=spformat(self._data))
        if self.metadata:
            right_str = ', metadata={})>'.format(self._metadata)
        else:
            right_str = ')>'
        return left_str + right_str

    def convert(self, datatype=None):
        """Converts self to a NxSeries of specified datatype.

        Destination datatype and self's datatype must have the same quantity.
        """
        quantity = self.datatype.quantity
        if quantity is None:
            return NotImplemented
        if datatype.quantity is not quantity:
            raise TypeError("Can only convert NxSeries within same quantity.")
        data = self._data.apply(partial(quantity.convert,
                                        unit=self.datatype.unit,
                                        target=datatype.unit))
        return NxSeries[datatype](data, **self.metadata)

    # Properties obtained from the underlying pandas Series.

    @property
    def index(self):
        """Pass-through of the pandas' index."""
        return self._data.index

    @property
    def at(self):
        """Returns NxData accessor for bracket-specified label position."""
        return _Accessor(self, self._data.at)

    @property
    def empty(self):
        return self._data.empty

    @property
    def iat(self):
        """Returns NxData accessor for bracket-specified sequence position."""
        return _Accessor(self, self._data.iat)

    @property
    def iloc(self):
        """Returns slicer for bracket-specified sequence position."""
        return _Accessor(self, self._data.iloc)

    @property
    def ix(self):
        """Returns slicer for bracket-specified sequence or label position."""
        return _Accessor(self, self._data.ix)

    @property
    def loc(self):
        """Returns slicer for bracket-specified sequence or label position."""
        return _Accessor(self, self._data.loc)

    @property
    def values(self):
        """Returns self._data.values, a numpy array."""
        return self._data.values
