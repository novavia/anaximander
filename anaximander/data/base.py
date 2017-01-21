#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the DataObject base archetype.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from anaximander.utilities import xprops
from anaximander.meta.nxtype import archetype
from anaximander.meta.nxobject import NxObject
from .exceptions import DataError

# =============================================================================
# DataObject class
# =============================================================================


@archetype
class DataObject(NxObject):
    """Archetype for all Data objects."""

    @xprops.weakproperty
    def context(self):
        """An optional context a Data object can refer to."""
        return None


class IndexedDataObject(DataObject):
    """Base class for NxSeries and NxFrame.

    Slicing relies on an index proxy class that has slightly different
    implementations for NxSeries and NxFrame. This is governed by the
    _index_proxy_type argument.
    """
    _index_proxy_type = None  # placeholder for concrete classes.

    def __getitem__(self, arg):
        return self._index_proxy_type(self, self._data).__getitem__(arg)

    @property
    def index(self):
        """Pass-through of the pandas' index."""
        return self._data.index

    @property
    def at(self):
        """Returns NxData accessor for bracket-specified label position."""
        return self._index_proxy_type(self, self._data.at)

    @property
    def empty(self):
        return self._data.empty

    @property
    def iat(self):
        """Returns NxData accessor for bracket-specified sequence position."""
        return self._index_proxy_type(self, self._data.iat)

    @property
    def iloc(self):
        """Returns slicer for bracket-specified sequence position."""
        return self._index_proxy_type(self, self._data.iloc)

    @property
    def ix(self):
        """Returns slicer for bracket-specified sequence or label position."""
        return self._index_proxy_type(self, self._data.ix)

    @property
    def loc(self):
        """Returns slicer for bracket-specified sequence or label position."""
        return self._index_proxy_type(self, self._data.loc)

    @property
    def values(self):
        """Returns self._data.values, a numpy array."""
        return self._data.values


class DataSlicingError(DataError):
    """Raised when a slice cannot be compelled into a DataObject."""
    pass
