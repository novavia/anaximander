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

from ..utilities import xprops
from ..utilities.functions import spformat
from ..utilities.nxtime import tz_aware
from ..utilities.nxrange import levels, Level, Levels
from ..meta import prototype, metacharacter, typeproperty
from .exceptions import DataError
from .base import IndexedDataObject, RowSlicer
from .data import NxData
from .plot import plot_series
from .annotations import interval, domain

__all__ = ['NxSeries']

# =============================================================================
# NxSeries prototype
# =============================================================================


class SeriesError(DataError):
    """Specialized exception for Series."""
    pass


@prototype
class NxSeries(IndexedDataObject, overtype=True, traits=(Sequence,)):
    """A read-only indexed sequence of NxData.

    NxSeries exposes a pandas Series, whose elements are of type
    self.datatype.dtype, i.e. the dtype of the NxDataType that is the
    metacharacter value of concrete NxSeries subtypes.
    """
    datatype = metacharacter(validate=lambda s: issubclass(s, NxData))

    @typeproperty
    def rowtype(cls):
        return cls.datatype

    @typeproperty
    def dtype(cls):
        return cls.datatype.dtype

    @typeproperty
    def quantity(cls):
        return cls.datatype.quantity

    @typeproperty
    def unit(cls):
        return cls.datatype.unit

    @property
    def domain(self):
        """The sequential domain for self.
        
        XXX: Note this is weak as it only works for non-empty series.
        """
        return domain(self.index)

    def __init__(self, data, index=None, context=None, ix_range=None,
                 **rgarg):
        """Creates an NxSeries instance.

        Params:
            data: any valid argument to pd.Series or an NxSeries object.
            index: any valid argument to index in pd.Series
            context: an optional context object.
            ix_range: any argument to the appropriate Range class intended
                to slice the index. For instance for a time series,
                ix_range can be a tuple that specifies a min and max time.
                For a categorical index, ix_range may be a single value
                or an iterable of admissiable levels.
            **rgarg: accepts a single keyword corresponding the index's name.
                if both ix_range and **rgarg are specified, a ValueError
                is raised.
        """
        # XXX: Eventually, add the possibility that what is passed to the
        # constructor is a Sequence of NxData. If this is done right, such
        # a collection (based on Sequence being a prototype) should carry
        # a data collectiveproperty, which results in the same data.data
        # assignment covering that use case.
        if isinstance(data, NxSeries):
            index = index or data.index
            context = context or data.context
            data = data.data
        series = pd.Series(data, index, self.datatype.dtype)
        series = tz_aware(series)
        if rgarg:
            if not list(rgarg.keys()) == [series.index.name]:
                msg = "Incorrect range argument passed to NxSeries."
                raise ValueError(msg)
            elif ix_range is not None:
                msg = "Specify either the index range or a named range \
                       argument but not both."
                raise ValueError(msg)
            else:
                ix_range = rgarg[series.index.name]
        self._data = self.subset(series, ix_range)
        self.context = context

    def __getitem__(self, key):
        """Slices self horizontally.

        This emulates pd.Series.__getitem__, where both integer and label
        keys are allowed. Note that if the index is integer-based, the
        method will behave like expected in Python, i.e. s[0:1] returns
        a series with a single item. This is in contrast to s.loc[0:1] which
        returns two.
        """
        return RowSlicer(self, self._data).__getitem__(key)

    def __repr__(self):
        base = '<NxSeries[{dt}]({content})>'
        return base.format(dt=self.datatype.__name__,
                           content=spformat(self._data))

    def __str__(self):
        return self._data.__str__()

    @classmethod
    def subset(cls, series, ix_range=None):
        """Subsets a valid series with supplied range argument."""
        if ix_range is None:
            return series
        if isinstance(series.index, pd.CategoricalIndex):
            range_ = levels(ix_range)
            if isinstance(range_, Level):
                series = series[range_ == series.index]
            elif isinstance(range_, Levels):
                series = series[series.index.isin(range_)]            
        else:
            range_ = interval(*ix_range, ref=series.index)
            lower, upper = range_.bounds
            series = series[(series.index >= lower) & (series.index <= upper)]
        return series

    def recast(self, data):
        """Returns updated version of self with a new pandas Series."""
        return type(self)(data, context=self.context)

    @property
    def empty(self):
        return self.data.empty

    @property
    def name(self):
        return self.data.name

    # Extension / contraction methods

    def extend(self, series):
        """Extends self with another NxSeries or pandas.Series."""
        if isinstance(series, NxSeries):
            series = series.data
        self._data = self._data.append(series)

    # Plotting functionalities

    @xprops.cachedproperty
    def curax(self):
        """Caches a matplotlib axis on which the series is plot."""
        return self.plot()

    def plot(self, ax='new', **kwargs):
        """Customized plot function.

        :param ax: 'new' for new plot, otherwise see super.
        """
        ax = plot_series(self, ax, **kwargs)
        self._curax = ax
        return ax

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
        return NxSeries[datatype](data, context=self.context)
