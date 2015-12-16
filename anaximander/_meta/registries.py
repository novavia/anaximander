#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements various object and type registries.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import abc
import weakref
from weakref import WeakSet, WeakValueDictionary

from blist import sorteddict

from ..arche import nxobject
from . import folios as fol
from ..utilities import xprops

#==============================================================================
### Registry base class
#==============================================================================


class NxRegistryBase(fol.NxRegistryABC):
    """Base class for registries and subregistries."""

    def __init__(self):
        self._root = self.__root__()
        self._root.parent = self

    @xprops.cachedproperty
    def root(self):
        """The root folio of the registry."""
        return None

    def _path(self, *args, **kwargs):
        """Forms a partial or complete path from *args, **kwargs.

        Kwargs are interpreted by looking up the __layers__ attribute
        of the class.
        """
        path = sorteddict(enumerate(args))
        for k, v in kwargs.items():
            try:
                ix = self.layernames.index[k]
            except ValueError:
                msg = "Incorrect keyword assignment {k}"
                raise ValueError(msg.lformat(msg))
            path[ix] = v
        return tuple(path.values())

    # Getters

    def _folio(self, *args, **kwargs):
        """Returns a Folio from a path specification.

        This function accepts truncated addresses but does not perform
        random search as subset would.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching path can be found.
        :returns: an NxFolio.
        """
        path = self._path(*args, **kwargs)
        return self.root.retrieve(*path)

    def branch(self, *args, **kwargs):
        """Returns a subregistry from a branch path specification.

        args and kwargs must unambiguously specify an existing path from the
        root, which may be truncated at an arbitrary depth.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching path can be found.
        :returns: a NxSubRegistry.
        """
        pass

    def subset(self, *args, **kwargs):
        """Returns a subregistry from a random access path search.

        Unlike branch, the arguments can provide partial path specifications
        and even point to multiple, separate branches.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :returns: a NxSubRegistry.
        """
        pass

    def addresses(self, *args, **kwargs):
        """Returns a list of registered addresses, subset by arguments.

        This function accepts truncated addresses but does not perform
        random search as subset would.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of addresses to objects, as tuples.
        """
        pass

    def values(self, *args, **kwargs):
        """Retuns an iterable of objects whose address match the arguments.

        This function accepts truncated addresses but does not perform
        random search as subset would.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of nxobject.
        """
        pass

    def items(self, *args, **kwargs):
        """Retuns an (address, obj) iterable where address match the arguments.

        This function accepts truncated addresses but does not perform
        random search as subset would.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of (address, nxobject) tuples.
        """
        pass

    def titles(self, *args, **kwargs):
        """Retuns an iterable of titles whose address match the arguments.

        This function accepts truncated addresses but does not perform
        random search as subset would.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of nxobject.
        """
        pass

    def browse(self, *args, **kwargs):
        """Retuns an iterable of titles whose address match the arguments.

        browse differs from titles in that it allows random address access,
        i.e. partial address specifications.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of nxobject.
        """

    def get(self, *args, **kwargs):
        """Returns a single title or raises an exception.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if the registry address doesn't exist.
        :raises ValueError: if the registry address points to multiple items.
        :returns: an nxobject.
        """
        pass

    def fetch(self, *args, **kwargs):
        """Returns a single title or raises an exception.

        fetch differs from get in that it allows random address access, i.e.
        partial address specifications.

        :param args: a partial sequential registry address specification.
        :param kwargs: a partial named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :raises ValueError: if the registry address points to multiple items.
        :returns: an nxobject.
        """
        pass

    def find(self, *args, **kwargs):
        """Returns an iterable of objects whose address supersedes the spec.

        :param args: a partial sequential registry address specification.
        :param kwargs: a partial named registry address specification.
        :returns: an nxobject.
        """
        pass

    # Admin

    def clear(self):
        """Clears the entire registry."""
        self.root.dispose()

    def __del__(self):
        """Upon dereferencing, the registry is cleared."""
        self.clear()

    def copy(self):
        """Returns a new registry that is a copy of self."""
        copy_ = type(self)()
        root = self.root.copy()
        copy_._root = root
        root.parent = copy_
        return copy_

    def hardcopy(self):
        """Returns a structure that copies self's content with hard refs."""
        pass

    def __repr__(self):
        pass

    def __str__(self):
        pass


class NxRegistry(NxRegistryBase):
    """Base class for registries.

    Class attributes include:
    * __root__: specification of the type of folio found at the root.
    * __layers__: accepts either a sequence of folio types, or a sequence of
    tuples of (<string>, <folio_type>), where the strings provide a name to
    each layer. Naming the layers allow passing kwargs to setter and getter
    functions.
    * __recurse__: accepts a PortFolio type that is used recursively beyond.
    specified layers.
    """
    __root__ = None  # Placeholder for the folio type of the root.
    __layers__ = ()  # Placeholder for the name and type of folio layers.
    __recurse__ = None  # Placeholder for a recursive folio type.

    @xprops.cachedproperty
    def layernames(self):
        try:
            return list(zip(*self.__layers__))[0]
        except (TypeError, IndexError):
            return []

    @xprops.cachedproperty
    def layertypes(self):
        try:
            return list(zip(*self.__layers__))[0]
        except (TypeError, IndexError):
            return list(self.__layers__)

    def layertype(self, depth=1, name=None):
        """Returns the layer type at a specified depth or name.

        :param depth: a layer depth whose type must be returned.
        :param name: a layer name. If layer name is specified it supersedes
        depth.
        :raises KeyError: if no determination can be made.
        :returns: an NxFolio type.
        """
        if name is not None:
            depth = self.layernames[name]
        try:
            return self.layertypes[depth]
        except IndexError:
            return self.__recurse__

    def register(self, obj, *args, **kwargs):
        """Open registration method, can be simplified in subclasses."""
        path = self._path(*args, **kwargs)
        for i in range(len(path) + 1, 0, -1):
            try:
                folio = self.root.retrieve(*path[:i])
            except KeyError:
                continue
            else:
                break
        else:
            i = 0
            folio = self.root
        types = (self.layertype(j) for j in range(i + 1, len(path) + 1))
        try:
            for k, tp in zip(path[i + 1:], types):
                insert = tp()
                folio[k] = insert
                folio = insert
        except TypeError:  # tp's call fails
            if isinstance(folio, fol.NxSchedule):
                folio.register(obj, k)
            else:
                msg = "Incorrect registration address specification."
                raise ValueError(msg)
        else:
            folio.register(obj)

    def unregister(self, obj, *args, **kwargs):
        """Unregisters object if found, otherwise silences exceptions."""
        path = self._path(*args, **kwargs)
        folio = self.root.retrieve(*path)
        if isinstance(folio, fol.NxSchedule):
            folio.unregister(obj, path[-1])
        else:
            folio.unregister(obj)


class Pool(NxRegistry):
    """A simple registry that provides membership functionality only."""
    __root__ = fol.NxPage

    def register(self, obj):
        self.root.add(obj)

    def unregister(self, obj):
        self.root.discard(obj)


class Hierarchy(NxRegistry):
    """A tree-like registry of individual objects, where keys are strings."""
    __root__ = fol.NxFolder
    __recurse__ = fol.NxFolder

    def register(self, obj, *args, **kwargs):
        path = self._path(*args, **kwargs)
        folio = self.__recurse__(obj)
        self.root.insert(folio, *path)

    def unregister(self, obj, *args, **kwargs):
        try:
            folio = self._folio(*args, **kwargs)
            folio.dispose()
        except KeyError:
            pass
