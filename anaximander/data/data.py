#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data module, which defines the Data archetype.

This version only supports univariate data types. Support for multi-variate
types based on compound numpy dtypes is planned for future release.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import numpy as np

from anaximander.utilities import functions as fun
from anaximander.meta.nxtype import archetype
from anaximander.meta.metadescriptors import TypeAttribute, metamethod
from .object import DataObject
from .quantities import Quantity

# =============================================================================
# Quantity class
# =============================================================================


def default_unit(cls):
    """Functions used as default for the unit TypeAttribute.

    If an NxDataType has a quantity, that quantity's measure is used by
    default. Otherwise unit defaults to None.
    """
    try:
        return cls.quantity.measure
    except AttributeError:
        return None


@archetype
class NxData(DataObject):
    """NxData are basic data containers for single measurements.

    Each NxDataType defines the following type attributes:
    * quantity: an optional Quantity instance that refers to the physical
        quantity described by the type. Defaults to None.
    * unit: a string used for representational purposes. If a quantity
        is defined and the unit is registered with that quantity, conversion
        functions are available. Defaults to None.
    * dtype: a numpy.dtype specification that indicates how instance values
        should be stored. Defaults to an unnamed float.
    * precision: an optional integer value used for printing instances if
        dtype is a float.
    """
    quantity = TypeAttribute(validate=fun.typechecker(Quantity))
    unit = TypeAttribute(default=default_unit, validate=fun.typechecker(str))
    dtype = TypeAttribute(default=np.dtype('float'))
    precision = TypeAttribute(validate=fun.typechecker(int))

    def __init__(self, data, **metadata):
        self._data = data
        self._metadata = metadata

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata

    @metamethod
    def __repr__(cls):
        """Repr method factory."""
        type_name = cls.__name__

        def inst_repr(self):
            left_str = '<{t}({d}'.format(t=type_name, d=self._data)
            if self.metadata:
                right_str = ', metadata={})>'.format(self._metadata)
            else:
                right_str = ')>'
            return left_str + right_str
        return inst_repr

    @metamethod
    def __str__(cls):
        """Print method factory."""
        if cls.dtype.kind is 'f' and cls.precision is not None:
            p = cls.precision
            dataformat = lambda d: '{:.{p}f}'.format(d, p=p)
        else:
            dataformat = str
        units = '' if cls.unit is None else ' ' + cls.unit

        def inst_str(self):
            return dataformat(self._data) + units

        return inst_str
