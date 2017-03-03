#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data module, which defines the Data archetypes.

This version only supports univariate data types. Support for multi-variate
types based on compound numpy dtypes is planned for future release.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import abc

import numpy as np

from anaximander.utilities import functions as fun
from anaximander.meta.nxtype import archetype
from anaximander.meta import typeattribute, metamethod, typeinitmethod
from .exceptions import ValidationError
from .base import DataObject
from .quantities import Quantity

__all__ = ['NxData', 'NxScalar', 'NxVector', 'NxFloat', 'NxBool',
           'NxInt', 'NxStr']

# =============================================================================
# NxData class and derived archetypes NxSclar and NxVector
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


class NxData(DataObject):
    """An abstract base type for NxData."""

    def __init__(self, data, context=None):
        if isinstance(data, NxData):
            context = context or data.context
            data = data._data
        self._data = self.cast(data)
        self.context = context

    @property
    def data(self):
        return self._data

    @abc.abstractclassmethod
    def cast(cls, value):
        """Casts data value into the appropriate type."""
        return NotImplemented


@archetype
class NxScalar(NxData):
    """Specializes NxData for scalar data types.

    Each NxScalar defines the following type attributes:
    * quantity: an optional Quantity instance that refers to the physical
        quantity described by the type. Defaults to None.
    * unit: a string used for representational purposes. If a quantity
        is defined and the unit is registered with that quantity, conversion
        functions are available. Defaults to None.
    * dtype: a numpy.dtype specification that indicates how instance values
        should be stored. Defaults to float64.
    * precision: an optional integer value used for comparing and printing
        instances. Default to 5, i.e. 1e-5 precision.
    """
    quantity = typeattribute(validate=fun.typecheck(Quantity))
    unit = typeattribute(default=default_unit, validate=fun.typecheck(str))
    dtype = typeattribute(default=np.dtype('float64'))
    precision = typeattribute(validate=fun.typecheck(int), default=5)

    @classmethod
    def cast(cls, value):
        """Converts an arbitrary value into the class' dtype."""
        try:
            return cls.dtype.type(value)
        except ValueError:
            raise ValidationError

    def __hash__(self):
        return hash(self._data)

    @metamethod
    def _compare(cls):
        """Comparison primitve for instances.

        Comparisons are based on the underlying data value alone. Evaluating
        equal does not imply that two instances share the same context.
        """
        if cls.quantity is not None:
            def convert(a, b):
                a = a.quantity.convert(a._data, a.unit)
                b = b.quantity.convert(b._data, b.unit)
                return (a, b)
        else:
            def convert(a, b):
                return (a._data, b._data)

        def extract(a, b):
            if a.quantity != b.quantity:
                raise TypeError("Cannot compare two different quantities.")
            return convert(a, b)
        return extract

    def __lt__(self, other):
        self_val, other_val = self._compare(other)
        if self.precision is not None:
            return self_val < other_val - 10 ** -self.precision
        else:
            return self_val < other_val

    def __le__(self, other):
        self_val, other_val = self._compare(other)
        if self.precision is not None:
            return self_val <= other_val + 10 ** -self.precision
        else:
            return self_val <= other_val

    def __eq__(self, other):
        self_val, other_val = self._compare(other)
        if self.precision is not None:
            return np.abs(self_val - other_val) <= 10 ** -self.precision
        else:
            return self_val == other_val

    def __ge__(self, other):
        self_val, other_val = self._compare(other)
        if self.precision is not None:
            return self_val >= other_val - 10 ** -self.precision
        else:
            return self_val >= other_val

    def __gt__(self, other):
        self_val, other_val = self._compare(other)
        if self.precision is not None:
            return self_val > other_val + 10 ** -self.precision
        else:
            return self_val > other_val

    def __ne__(self, other):
        self_val, other_val = self._compare(other)
        if self.precision is not None:
            return np.abs(self_val - other_val) > 10 ** -self.precision
        else:
            return self_val != other_val

    def convert(self, datatype=None):
        """Converts self to a NxScalar of specified type.

        The destination datatype must have the same quantity.
        """
        if self.quantity is None:
            return NotImplemented
        if datatype.quantity is not self.quantity:
            raise TypeError("Can only convert NxScalar within same quantity.")
        data = self.quantity.convert(self._data, self.unit, datatype.unit)
        return datatype(data, self.context)

    # Numeric types coercion functions

    def __complex__(self):
        return complex(self._data)

    def __int__(self):
        return int(self._data)

    def __float__(self):
        return float(self._data)

    def __round__(self):
        return round(self._data)

    @typeinitmethod
    def __coerce_index__(cls):
        """Adds __index__ coercion method based on dtype."""
        if cls.dtype.kind in ('b', 'i', 'u'):
            cls.__index__ = lambda s: int(s._data)

    def __repr__(self):
        type_name = type(self).__name__
        return '<{t}({d})>'.format(t=type_name, d=self._data)

    def __str__(self):
        units = '' if self.unit is None else ' ' + self.unit
        return '{s._data:.{s.precision}}'.format(s=self) + units


NxFloat = NxScalar.subtype(name='NxFloat', dtype=np.dtype('float'))
NxInt = NxScalar.subtype(name='NxInt', dtype=np.dtype('int'))
NxBool = NxScalar.subtype(name='NxBool', dtype=np.dtype('bool'))
NxStr = NxScalar.subtype(name='NxStr', dtype=np.dtype('str'))


class NxVector(NxData):
    """Specializes NxData for vectorial data types."""
    pass
