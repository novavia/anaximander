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


@prototype
class NxSeries(DataObject, overtype=True):
    """A read-only indexed sequence of NxData.

    NxSeries exposes a pandas Series, whose elements are of type
    self.datatype.dtype, i.e. the dtype of the NxDataType that is the
    metacharacter value of concrete NxSeries subtypes.
    """

    datatype = MetaCharacter(validate=lambda s: issubclass(s, NxData))

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
