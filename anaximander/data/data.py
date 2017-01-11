#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data module, which defines the Data archetype.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from anaximander.utilities import functions as fun
from anaximander.meta.nxtype import archetype
from anaximander.meta.metadescriptors import TypeAttribute
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
    """
    quantity = TypeAttribute(validate=fun.typechecker(Quantity))
    unit = TypeAttribute(default=default_unit, validate=fun.typechecker(str))
    dtype = TypeAttribute()
