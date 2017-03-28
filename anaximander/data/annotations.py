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

import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.palettes import _ColorPalette as ColorPalette

from ..utilities.functions import typecheck
from ..utilities import xprops
from ..utilities.nxtime import datetime
from ..utilities import nxattr
from ..utilities.nxrange import float_interval, time_interval, \
    string_interval
from ..meta.metadescriptors import MetaInstance
from ..meta import NxObject, archetype, typeattribute, typeproperty, \
    typeinitmethod, typemethod
from .base import DataObject
from .data import NxScalar
from .fields import Field, Raw, Nested, Dict, List, String, UUID, \
    Number, Integer, Decimal, Boolean, FormattedString, Float, DateTime, \
    LocalDateTime, Time, Date, TimeDelta, Url, URL, Email, Method, Function, \
    Str, Bool, Int, Constant, Scalar, Timestamp, Duration, Period

__all__ = []


PALETTE = sns.color_palette("muted")

# =============================================================================
# Selection domain functions, extending nxrange
# =============================================================================

# Mapping from field type to selection domain
_field_domain = {Field: None,
                 Raw: None,
                 Nested: None,
                 Dict: None,
                 List: None,
                 String: 'string',
                 UUID: 'string',
                 Number: 'float',
                 Integer: 'float',
                 Decimal: 'float',
                 Boolean: 'float',
                 FormattedString: 'string',
                 Float: 'float',
                 DateTime: 'time',
                 LocalDateTime: 'time',
                 Time: 'time',
                 Date: 'time',
                 TimeDelta: None,
                 Url: 'string',
                 URL: 'string',
                 Email: 'string',
                 Method: None,
                 Function: None,
                 Str: 'string',
                 Bool: 'float',
                 Int: 'float',
                 Constant: None,
                 Timestamp: 'time',
                 Duration: None,
                 Period: 'time'
                 }

# Mapping from numpy dtype kinds to selection domain
_kind_domain = {'b': 'float',
                'i': 'float',
                'u': 'float',
                'f': 'float',
                'c': None,  # complex floating-point
                'm': None,  # timedelta
                'M': 'time',
                'O': None,  # Object
                'S': None,  # (byte-)string
                'U': 'string',  # Unicode
                'V': None,  # void
                }

# Maps index types to the appropriate selection domain
_index_domain = {pd.Int64Index: 'float',
                 pd.Float64Index: 'float',
                 pd.DatetimeIndex: 'time'}


def _kind_to_domain(kind):
    """Returns a domain from a numpy dtype.kind."""
    try:
        return _kind_domain[kind]
    except KeyError:
        return None


def _fieldtype_to_domain(field_type):
    """Returns a domain from a field type."""
    if field_type.__interval__ is not None:
        return field_type.__interval__.__func__
    for ft in field_type.__mro__:
        try:
            return _field_domain[ft]
        except KeyError:
            pass
    return None


def _indextype_to_domain(index_type):
    """Returns an interval helper function from a pandas index type."""
    for it in index_type.__mro__:
        try:
            return _index_domain[it]
        except KeyError:
            pass
    return None    


def domain(ref):
    """Returns a selection domain based on ref.

    ref accepts a wide range of possible values:
        * A numpy dtype kind
        * A numpy dtype
        * A data.field or field type
        * A data.NxScalar instance or type
        * A pandas index or index type
    """
    if isinstance(ref, type):
        if issubclass(ref, Field):
            return _fieldtype_to_domain(ref)
        elif issubclass(ref, NxScalar):
            return _kind_to_domain(NxScalar.dtype.kind)
        elif issubclass(ref, pd.Index):
            return _indextype_to_domain(ref)
    elif isinstance(ref, np.dtype):
        return _kind_to_domain(ref.kind)
    elif isinstance(ref, str):
        if ref in ('float', 'string', 'time'):
            return ref
        return _kind_to_domain(ref)
    elif isinstance(ref, Field):
        if isinstance(ref, Scalar):
            kind = ref.datatype.dtype.kind
            return _kind_to_domain(kind)
        else:
            return _fieldtype_to_domain(type(ref))
    elif isinstance(ref, NxScalar):
        return _kind_to_domain(ref.dtype.kind)
    elif isinstance(ref, pd.Index):
        return _indextype_to_domain(type(ref))
    else:
        return None


# Maps domain to interval helper function
_domain_interval = {'float': float_interval,
                    'string': string_interval,
                    'time': time_interval}


def _domain_to_interval(domain):
    return _domain_interval.get(domain, NotImplemented)


def interval(lower=None, upper=None, *, ref):
    """Returns an interval helper function based on ref keyword-only argument.

    ref accepts a wide range of possible values:
        * A numpy dtype kind
        * A numpy dtype
        * A data.field or field type
        * A data.NxScalar instance or type
        * A pandas index or index type
    """
    dom = domain(ref)
    func = _domain_to_interval(dom)
    if func is NotImplemented:
        msg = "No interval function is provided for {0}"
        raise ValueError(msg.format(ref))
    return func(lower, upper)


# Maps domain to location function
_domain_location = {'float': float,
                    'string': str,
                    'time': datetime}


def _domain_to_location(domain):
    return _domain_location.get(domain, NotImplemented)


def location(loc, *, ref):
    """Returns a properly formatted location from loc based on ref.

    ref accepts a wide range of possible values:
        * A numpy dtype kind
        * A numpy dtype
        * A data.field or field type
        * A data.NxScalar instance or type
        * A pandas index or index type
    """
    dom = domain(ref)
    func = _domain_to_location(dom)
    if func is NotImplemented:
        msg = "No location function is provided for {0}"
        raise ValueError(msg.format(ref))
    return func(loc)        

# =============================================================================
# Pen base class for Marker and Highlighter
# =============================================================================


class PenError(Exception):
    """Specialized exception type for Pens."""
    pass


class Shade(MetaInstance):
    """Specialized MetaInstance declarator for Pen shades."""

    def __init__(self, **plargs):
        """Takes plot arguments for instantiation."""
        super().__init__(use_name='shade', inherit=True, **plargs)


def shade(**plargs):
    """Helper function used in Pen type declarations."""
    return Shade(**plargs)


@archetype
class Pen(NxObject):
    """Archetype for Pen types."""
    palette = typeattribute(default=PALETTE, validate=typecheck(ColorPalette))

    @typeproperty
    def shades(cls):
        return cls.__shades__.copy()

    def __init__(self, shade, register=True, **plargs):
        """Creates a new marker instance.

        Params:
            shade: a string used for representational purposes, and optionally
                tor register the marker as a shades in its type.
            register: whether to register the instance as a shade or not.
            **plargs: plot arguments used in figures.
        """
        self.shade = shade
        self.plargs = plargs
        if register:
            self._register()

    def _register(self):
        """Registers self as a shade."""
        self.__shades__[self.shade] = self

    def _unregister(self):
        """Unregisters self as a shade."""
        try:
            del self.__shades__[self.shade]
        except KeyError:
            pass

    @typeinitmethod
    def _set_shades(cls):
        """Creates the __shades__ attribute."""
        cls.__shades__ = OrderedDict()

    @typemethod
    def __call__(cls, shade, register=True, **plargs):
        """Customizes the metaclass __call__ to fetch from instance cache."""
        if isinstance(shade, cls):
            return shade
        try:
            return cls.__shades__[shade]
        except KeyError:
            instance = cls.__new__(cls, shade, register, **plargs)
            cls.__init__(instance, shade, register, **plargs)
            return instance


class Marker(Pen):
    """Base class for all Marker types."""
    blank = shade(color='white')


class Highlighter(Pen):
    """Base class for all Highlighter types."""
    blank = shade(color='white')

# =============================================================================
# Marks and Highlights
# =============================================================================


class DataAnnotation(NxObject):
    """Base class for Marks and Highlights."""

    @xprops.typedweakproperty(DataObject)
    def dataobject(self):
        """A weak reference to the target data object."""
        return None

    def __init__(self, dataobject):
        """Requires a DataObject and a Marker instance."""
        self.dataobject = dataobject

    @abc.abstractproperty
    def data(self):
        """Returns a dataobject subset that matches self's scope."""
        return None


@nxattr.s(init=False)
class Mark(DataAnnotation):
    """An annotation with a location on a sequential axis."""
    dataobject = nxattr.ib()
    marker = nxattr.ib()
    location = nxattr.ib()

    def __init__(self, dataobject, marker, loc):
        super().__init__(dataobject)
        self.marker = marker
        self.domain = domain(dataobject.index)
        self.location = location(loc, ref=self.domain)

    @property
    def ix(self):
        """The index position closest to self.location in dataobject."""
        return self.dataobject.index.get_loc(self.location, method='nearest')

    @property
    def dataloc(self):
        """The location closest to self.location in dataobject.index."""
        return self.dataobject.index[self.ix]

    @property
    def data(self):
        """Extracts data from dataobject at self's location.

        This returns either a NxData if dataobject is a series, or an
        NxRecord if dataobject is a dataframe.
        """
        return self.dataobject.iloc[self.ix]

    @property
    def shade(self):
        return self.marker.shade


@nxattr.s(init=False)
class Highlight(DataAnnotation):
    """An annotation with a lower and upper location."""
    dataobject = nxattr.ib()
    highlighter = nxattr.ib()
    interval = nxattr.ib()

    def __init__(self, dataobject, highlighter, lower=None, upper=None):
        super().__init__(dataobject)
        self.highlighter = highlighter
        self.domain = domain(dataobject.index)
        self.interval = interval(lower, upper, ref=self.domain)

    @property
    def lower(self):
        return self.interval.lower

    @property
    def upper(self):
        return self.interval.upper

    @property
    def length(self):
        return self.interval.length

    @property
    def bounds(self):
        return self.interval.bounds

    @property
    def shade(self):
        return self.highlighter.shade

    @property
    def data(self):
        """Returns the slice of dataobject corresponding to self's bounds."""
        return self.dataobject.loc[self.lower:self.upper]

    def transitions(self, prev='blank', next='blank'):
        """Returns the lower and upper transitions.

        Highlighters before and after the highlight can be specified with
        prev and next, both defaulting to the blank highligher.
        """
        highlighter_type = type(self.highlighter)
        prev_shade = highlighter_type(prev).shade
        next_shade = highlighter_type(next).shade
        this_shade = self.highlighter.shade
        return ((self.lower, prev_shade, this_shade),
                (self.upper, this_shade, next_shade))
