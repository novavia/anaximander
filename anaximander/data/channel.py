#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The channel module defines the base data I/O facilities.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================


import abc

from anaximander.utilities import nxattr, xprops
from anaximander.meta.nxobject import NxObject
from anaximander.meta.metadescriptors import typeproperty
from .tract import DataTract

# =============================================================================
# DataChannel class
# =============================================================================


@nxattr.s
class DataChannel(NxObject):
    """An abstract base class for holding process-storage connections.

    A DataChannel requires an instance of a DataTract, and a store, which is
    any kind of object that sufficiently describes a data storage resource,
    e.g. a database table or a file.
    A DataChannel provides methods for loading / dumping data from / to
    a storage unit. Note that this is an abstract construct, i.e.
    instantiating a DataChannel does not actually create any kind of
    physical channel.
    The virtue of the DataChannel is to provide a unified interface between
    Anaximander DataObjects and persistent storage.
    """
    tract = nxattr.ib(validator=nxattr.validators.instance_of(DataTract))
    store = nxattr.ib()

    @property
    def frametype(self):
        """The default NxDataFrame subtype for loading datasets."""
        return self.tract.Frame


class DataOperator(NxObject):
    """Primitive for DataLoader and DataDumper."""

    def __init__(self, channel):
        self.channel = channel

    @property
    def tract(self):
        return self.channel.tract


class DataLoader(NxObject):
    """A representation of a pending data loading operation.

    DataLoader serves as a base class for concrete loading operations such
    as database table queries and fetching data from files.
    DataLoader object only stock references to loading protocol and data
    selection criteria, until their __call__ method is called, which
    tries to execute the loading operation and generate results in the form
    of one or more DataObjects.
    A DataLoader requires a DataChannel to be instantiated. Beyond then,
    various methods exist to return more precise DataLoaders by composition
    of attributes. This works similarly to, say, SQLAlchemy Query object.
    """

    def __init__(self, channel):
        self.channel = channel

    @xprops.cachedproperty
    def data(self):
        return None

    @abc.abstractmethod
    def __run__(self):
        """Runs and returns raw results of the underlying loading operation."""
        pass

    def run(self):
        """Executes the query and caches the raw data."""
        self._data = self.__run__()

    def __call__(self, frametype=None):
        """Returns the data in the appropriate NxDataFrame subtype.

        By default, the return value's type is self.channel.frametype.
        However the default can be overriden by passing an explicit frametype.
        """
        frametype = frametype or self.channel.frametype
        if self.data is None:
            self.run()
        return frametype(self.data)


class DataDumper(NxObject):
    """A representation of a pending data dumping operation.

    This is the counterpart of DataLoader for dumping data to persistent
    storage through a channel.
    """

    def __init__(self, channel):
        self.channel = channel

    def __call__(self):
        """Dumps results."""
        pass
