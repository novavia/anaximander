#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantities module, which defines a class for physical quantities.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from ..utilities import nxattr
from ..meta import directory, NxObject
from .exceptions import DataError

__all__ = ['Quantity']

# =============================================================================
# Quantity class
# =============================================================================


class UnitError(DataError):
    """Customized Exception class for errors associated with Quantity units."""
    pass


@nxattr.s
class Quantity(NxObject, registry=directory('name')):
    """A class that holds a name and a reference unit name or 'measure'.

    The class also registers additional unit names along with their
    value expressed in units of the measure.
    """
    name = nxattr.ib(validator=nxattr.validators.instance_of(str))
    measure = nxattr.ib(validator=nxattr.validators.instance_of(str))

    def __attrs_post_init__(self):
        # Create a dictionay to hold unit names and conversion rates.
        # _units can also take callables as values, in which case the
        # argument should be a number in the registered unit, and the result
        # its equivalent number in the reference unit.
        self._units = {self.measure: 1.}

    @property
    def units(self):
        """Returns a copy of self's unit registry -i.e. read-only."""
        return dict(self._units)

    def register_unit(self, name, convert, reverse=None):
        """Registers a named unit, along with its value in reference units.

        Params:
            name: the name of the unit to register.
            convert: the value of the registered unit in the Quantity's
                reference measure. Alternatively, a conversion function to
                go from the registered unit to the reference unit.
            reverse: optional. If a callable is passed to convert, then
                reverse enables conversions from the reference unit to the
                registered unit. It is stored under the name '_u' where 'u'
                would be the name of the unit. If convert is not callable,
                reverse is ignored.
        """
        if callable(convert):
            if reverse is None:
                msg = "A unit that uses a conversion function must define " + \
                    "a reverse conversion function as well."
                raise UnitError(msg)
            else:
                self._units['_' + name] = reverse
        self._units[name] = convert

    def deregister_unit(self, name):
        """Deregisters unit."""
        try:
            del self._units[name]
        except KeyError:
            pass
        try:
            del self._units['_' + name]
        except KeyError:
            pass

    def _converter(self, unit=None):
        """Returns a conversion function for the supplied unit.

        The return function converts from unit to self's measure.
        """
        if unit is None:
            return lambda v: v
        try:
            unit_conversion = self._units[unit]
        except KeyError:
            msg = "Invalid unit name {0} passed to convert"
            raise UnitError(msg.format(unit))
        if callable(unit_conversion):
            return unit_conversion
        else:
            return lambda v: v * unit_conversion

    def _reverse_converter(self, unit=None):
        """Returns a reverse conversion function for the supplied unit.

        The return function converts from self's measure to the unit.
        """
        if unit is None:
            return lambda v: v
        try:
            unit_conversion = self._units[unit]
        except KeyError:
            msg = "Invalid unit name {0} passed to convert"
            raise UnitError(msg.format(unit))
        if callable(unit_conversion):
            return self._units['_' + unit]
        else:
            return lambda v: v / unit_conversion

    def convert(self, value, unit=None, target=None):
        """Converts value in 'unit' into the 'target' unit.

        Params:
            value: a numerical value.
            unit (str): a string found in the unit registry. If None,
                defaults to self's measure.
            target (str): a string found in the unit registry, into which
                the value will be converted. If None, this is self's measure.
        Raises:
            UnitError: if unit or target aren't registered.
        Returns:
            the converted value.
        """
        unit_converter = self._converter(unit)
        if target is None:
            return unit_converter(value)
        target_converter = self._reverse_converter(target)
        return target_converter(unit_converter(value))

    def iterconvert(self, iterable, unit=None, target=None):
        """Converts iterable of numerical values from 'unit' to 'target' unit.

        Params:
            iterable: an iterable of numerical values.
            unit (str): a string found in the unit registry.
            target (str): a string found in the unit registry, into which
                the value will be converted. If None, this is self's measure.
        Raises:
            UnitError: if unit or target aren't registered.
        Returns:
            a generator of converted values.
        """
        unit_converter = self._converter(unit)
        if target is None:
            return (unit_converter(v) for v in iterable)
        target_converter = self._reverse_converter(target)
        return (target_converter(unit_converter(v)) for v in iterable)
