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
from .base import IndexedDataObject, RowSlicer
from .data import NxData

# =============================================================================
# NxSeries prototype
# =============================================================================


@prototype
class NxSeries(IndexedDataObject, overtype=True, traits=(Sequence,)):
    """A read-only indexed sequence of NxData.

    NxSeries exposes a pandas Series, whose elements are of type
    self.datatype.dtype, i.e. the dtype of the NxDataType that is the
    metacharacter value of concrete NxSeries subtypes.
    """
    datatype = MetaCharacter(validate=lambda s: issubclass(s, NxData))

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

    def __init__(self, data, index=None, context=None):
        # XXX: Eventually, add the possibility that what is passed to the
        # constructor is a Sequence of NxData. If this is done right, such
        # a collection (based on Sequence being a prototype) should carry
        # a data collectiveproperty, which results in the same data.data
        # assignment covering that use case.
        if isinstance(data, NxSeries):
            index = index or data.index
            context = context or data.context
            data = data.data
        self._data = pd.Series(data, index, self.datatype.dtype, copy=True)
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
