#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines data annotations.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements and constants
# =============================================================================

import abc
from collections import OrderedDict

import seaborn as sns
from seaborn.palettes import _ColorPalette as ColorPalette

from ..utilities.functions import typecheck
from ..utilities import xprops
from ..meta.metadescriptors import MetaInstance
from ..meta import NxObject, archetype, typeattribute, typeproperty, \
    typeinitmethod, typemethod
from .base import DataObject


__all__ = []


PALETTE = sns.color_palette("muted")

# =============================================================================
# Marker base class
# =============================================================================


class MarkerError(Exception):
    """Specialized exception type for Markers."""
    pass


class Shade(MetaInstance):
    """Specialized MetaInstance declarator for Marker shades."""

    def __init__(self, **plargs):
        """Takes plot arguments for instantiation."""
        super().__init__(inherit=True, use_name=True, **plargs)


def shade(**plargs):
    """Helper function used in Marker type declarations."""
    return Shade(**plargs)


@archetype
class Marker(NxObject):
    """Archetype for Marker types."""
    palette = typeattribute(default=PALETTE, validate=typecheck(ColorPalette))

    @typeproperty
    def shades(cls):
        return cls.__shades__.copy()

    def __init__(self, name, register=True, **plargs):
        """Creates a new marker instance.

        Params:
            name: a string used for representational purposes, and optionally
                tor register the marker as a shades in its type.
            register: whether to register the instance as a shade or not.
            **plargs: plot arguments used in figures.
        """
        self.name = name
        self.plargs = plargs
        if register:
            self._register()

    def _register(self):
        """Registers self as a shade."""
        self.__shades__[self.name] = self

    def _unregister(self):
        """Unregisters self as a shade."""
        try:
            del self.__shades__[self.name]
        except KeyError:
            pass

    @typeinitmethod
    def _set_shades(cls):
        """Creates the __shades__ attribute."""
        cls.__shades__ = OrderedDict()

    @typemethod
    def __call__(cls, name, register=True, **plargs):
        """Customizes the metaclass __call__ to fetch from instance cache."""
        try:
            return cls.__shades__[name]
        except KeyError:
            instance = cls.__new__(cls, name, register, **plargs)
            cls.__init__(instance, name, register, **plargs)
            return instance


# =============================================================================
# Marks and Highlights
# =============================================================================


class DataAnnotation(NxObject):
    """Base class for Marks and Highlights."""

    @xprops.typedweakproperty(DataObject)
    def dataobject(self):
        """A weak reference to the target data object."""
        return None

    @xprops.typedweakproperty(Marker)
    def marker(self):
        """A weak reference to a marker instance."""
        return None

    def __init__(self, dataobject, marker):
        """Requires a DataObject and a Marker instance."""
        self.dataobject = dataobject
        self.marker = marker

    @abc.abstractproperty
    def data(self):
        """Returns a dataobject subset that matches self's scope."""
        return None


class Mark(DataAnnotation):
    """An annotation with a location on a sequential axis."""

    def __init__(self, dataobject, marker, location):
        super().__init__(dataobject, marker)
        self.location = location

    # TODO:
        # * Make a data property. It returns either a Record or a value.
        # Make it a placeholder for now, as doing interpolation systematically
        # will be a bit time-consuming.


class Highlight(DataAnnotation):
    """An annotation with a lower and upper location."""

    def __init__(self, dataobject, marker, lower, upper):
        super().__init__(dataobject, marker)
        self.lower = lower
        self.upper = upper

    # TODO:
        # * Make an interval property. It needs to look up the kind of
        # of index of dataobject (either index of a series, or sequential
        # key of a frame) to return the appropriate interval type.
        # * Create a data property, which returns a subset of dataobject
        # after applying the range arguments.