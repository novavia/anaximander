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

import abc
from collections.abc import Sequence

import pandas as pd

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

    @abc.abstractproperty
    def data(self):
        """Returns the underlying data for a DataObject.

        For an NxSeries or NxDataFrame, it is the exposed pandas Series or
        DataFrame. For an NxRecord, this makes a pandas Series whose labels
        are the record's attributes. For an NxData, this returns either
        a numpy scalar (NxScalar) or a numpy structured array (NxVector).
        """
        return NotImplemented

    @xprops.weakproperty
    def context(self):
        """An optional context a Data object can refer to."""
        return None


class DataSlicingError(DataError):
    """Raised when a slice cannot be compelled into a DataObject."""
    pass


class RowSlicer(object):
    """Wraps pandas indexer object to return DataObjects.

    The class serves to emulate the behavior of Pandas' iloc and loc
    selectors. However, it is explicitly restricted to selecting horizontal
    slices, i.e. either single rows (returning an NxData for a NxSeries,
    or an NxRecord for a NxDataFrame), or subsets of rows (in which case
    the result's type is the same as the caller.)
    """

    def __init__(self, nxdata, pdidx):
        """Instantiated with an IndexedDataObject and a pandas accessor.

        The pandas accessor is expected to be ix, iloc or loc.
        """
        self._nxdata = nxdata
        self._pdidx = pdidx

    @property
    def selector(self):
        """Returns 'ix', 'iloc', or 'loc' to provide underlying semantic."""
        # Returns _IXIndexer, _LocIndexer or _iLocIndexer
        type_name = type(self._pdidx).__name__
        return type_name[1:-7].lower()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            msg = "'{}' selector on an IndexedDataObject can only be used " + \
                "for row slicing and does not admit tuple key arguments."
            raise DataSlicingError(msg.format(self.selector))
        pdreturn = self._pdidx.__getitem__(key)
        context = self._nxdata.context
        # case series -> series or dataframe -> dataframe
        if isinstance(pdreturn, type(self._nxdata._data)):
            return type(self._nxdata)(pdreturn, context=context)
        # else pdreturn is a single row
        elif isinstance(pdreturn, pd.Series):
            return self._nxdata.rowtype(**pdreturn, context=context)
        else:
            return self._nxdata.rowtype(pdreturn, context=context)


class IndexedDataObject(DataObject):
    """Base class for NxSeries and NxDataFrame.

    Both NxSeries and NxDataFrame have indexed rows and this base class
    governs iteration behavior and row selection.
    """
    rowtype = None  # placeholder for concrete classes.

    @property
    def data(self):
        return self._data.copy()

    def __len__(self):
        return self._data.__len__()

    @property
    def index(self):
        """Pass-through of the pandas' index."""
        return self._data.index

    @property
    def empty(self):
        return self._data.empty

    @property
    def iloc(self):
        """Returns row slicer for bracket-specified sequence position."""
        return RowSlicer(self, self._data.iloc)

    @property
    def loc(self):
        """Returns row slicer for bracket-specified label position."""
        return RowSlicer(self, self._data.loc)
