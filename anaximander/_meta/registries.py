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
from ..utilities import functions as fun, xprops

#==============================================================================
### Registry base classes
#==============================================================================


class NxRegistryBase(fol.NxRegistryABC):
    """Base class for registries and subregistries."""

    def __init__(self):
        self._root = self.__root__()

    @property
    def root(self):
        """The root folio of the registry."""
        return self._root

    @property
    def _root(self):
        return getattr(self, '_root_cache', None)

    @_root.setter
    def _root(self, folio):
        if self._root is not None:
            del self._root_cache.parent
            self._root_cache = None
        if folio is not None:
            folio.parent = self
            self._root_cache = folio

    @_root.deleter
    def _root(self):
        if self._root is not None:
            del self._root_cache.parent
            self._root_cache = None

    def _path(self, *args, **kwargs):
        """Forms a partial or complete path from *args, **kwargs.

        Kwargs are interpreted by looking up the __layers__ attribute
        of the class.
        """
        path = sorteddict(enumerate(args))
        for k, v in kwargs.items():
            try:
                ix = self.layernames.index(k)
            except ValueError:
                msg = "Incorrect keyword assignment {k}"
                raise ValueError(fun.lformat(msg))
            path[ix] = v
        return tuple(path.values())

    # Getters

    def __getitem__(self, key):
        """Convenience function to bypass .root calls."""
        return self.root[key]

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
        path = self._path(*args, **kwargs)
        if not path:
            return NxSubRegistry(self, copy=True)
        subregistry = NxSubRegistry(self)
        origin, copy = self.root, subregistry.root
        try:
            while path:
                key, *path = path
                if path:
                    if isinstance(origin, fol.NxPortFolio):
                        copy[key] = origin[key].proxy()
                        origin = origin[key]
                        copy = copy[key]
                    else:
                        raise KeyError
                else:
                    copy[key] = origin[key].copy()
        except KeyError:
            msg = "No branch matching the path specification."
            raise KeyError(msg)
        return subregistry

# FIXME: this may still be incorrect in the sense that if kwargs are specified,
# these shouldn't be used at other locations on the path.
    def subset(self, *args, **kwargs):
        """Returns a subregistry from a random access path search.

        Unlike branch, the arguments can provide partial path specifications
        and even point to multiple, separate branches.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises ValueError: if the path specification uses incorrect keywords.
        :returns: a NxSubRegistry.
        """
        path = self._path(*args, **kwargs)
        if not path:
            return NxSubRegistry(self, copy=True)
        if not isinstance(self.root, fol.NxPortFolio):
            return NxSubRegistry(self, empty=True)
        subregistry = NxSubRegistry(self)
        folios = self.root.search(*path)
        for f in folios:
            subregistry.root.insert(f.copy(), *f.path)
        return subregistry

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
        del self._root

    def __del__(self):
        """Upon dereferencing, the registry is cleared."""
        self.clear()

    def copy(self):
        """Returns a new registry that is a copy of self."""
        copy_ = type(self)()
        root = self.root.deepcopy()
        copy_._root = root
        return copy_

    def hardcopy(self):
        """Returns a structure that copies self's content with hard refs."""
        return self.root.hardcopy()

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
    functions. By convention, if the last layer is of type NxSchedule,
    the kwarg 'key' will be recognized as designating a schedule's key.
    * __recurse__: accepts a PortFolio type that is used recursively beyond.
    specified layers.
    """
    __root__ = None  # Placeholder for the folio type of the root.
    __layers__ = ()  # Placeholder for the name and type of folio layers.
    __recurse__ = None  # Placeholder for a recursive folio type.

    @xprops.cachedproperty
    def layernames(self):
        try:
            bottom_layer_type = self.layertypes[-1]
        except KeyError:
            bottom_layer_type = self.__root__ if not self.__recurse__ else None
        add_key = bottom_layer_type == fol.NxSchedule
        try:
            names = list(list(zip(*self.__layers__))[0])
        except (TypeError, IndexError):
            names = []
        if add_key:
            names.append('key')
        return tuple(names)

    @xprops.cachedproperty
    def layertypes(self):
        try:
            return tuple(zip(*self.__layers__))[1]
        except (TypeError, IndexError):
            return tuple(self.__layers__)

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
        """Open registration method, can be simplified in subclasses.

        :param obj: an nxobject.
        :param *args, **kwargs: an address specification.
        :raises ValueError: if the address specification is incorrect.
        :returns True: if registration is successful.
        """
        err_msg = "Incorrect registration address specification."
        path = self._path(*args, **kwargs)
        folio = self.root
        i = -1  # Layer counter.
        while path:
            if isinstance(folio, fol.NxPortFolio):
                key, *path = path
                i += 1
                try:
                    folio = folio[key]
                except KeyError:
                    try:
                        insert = self.layertype(i)()
                        folio[key] = insert
                        folio = insert
                    # Failure to create insert or incorrect key type.
                    except (KeyError, TypeError):
                        raise ValueError(err_msg)
            else:
                break
        else:
            try:
                folio.register(obj)
                return True
            except TypeError:
                raise ValueError(err_msg)
        try:
            folio.register(obj, *path)
            return True
        except TypeError:
            raise ValueError(err_msg)

    def unregister(self, obj, *args, **kwargs):
        """Unregisters object if found, otherwise silences exceptions."""
        path = self._path(*args, **kwargs)
        try:
            folio = self.root.retrieve(*path)
            return folio.unregister(obj)
        except KeyError:  # Assume an NxSchedule terminal folio.
            folio = self.root.retrieve(*path[:-1])
            return folio.unregister(obj, path[-1])
        except TypeError:  # NxSchedule missing key arg in unregister
            return False


class NxSubRegistry(NxRegistryBase):
    """An accessor object to a subset of an NxRegistry."""

    def __init__(self, registry, *, copy=False, empty=False):
        """Initializes a subregistry from registry.

        :param registry: an NxRegistry.
        :param copy: if True, the subregistry points to the entire registry.
        :param empty: if True, the subregistry's root is not even a proxy
            to the registry's root, but rather a new instance of same type.
        """
        self._registry = registry
        if copy:
            self._root = registry.root.copy()
        elif empty:
            self._root = registry.__root__()
        else:
            self._root = registry.root.proxy()

    @xprops.cachedproperty
    def registry(self):
        """The original registry."""
        return None

    @property
    def layernames(self):
        return self.registry.layernames

    @property
    def layertypes(self):
        return self.registry.layertypes


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
