#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements various object and type registries.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

from . import folios as fol
from ..utilities import functions as fun, xprops

__all__ = ['RegistrationError', 'NxRegistry', 'Pool', 'Hierarchy',
           'Tree', 'Book', 'Roll', 'Directory']

# =============================================================================
# Registry base classes
# =============================================================================


# TODO: may replace some of the standard exceptions used in this module.
class RegistrationError(Exception):
    """A customize Exception type for registries."""
    pass


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

    @fun.args_or_kwargs
    def _path(self, *args, **kwargs):
        """Forms a partial or complete path from *args or **kwargs.

        _path accepts either arguments or keyword arguments but not
        both.
        Kwargs are interpreted by looking up the layers attribute
        of the registry.
        The arguments can form a partial path from the root to a folio
        contained in the registry, but cannot ommit intermediary keys.
        """
        if args:
            return args
        path = []
        try:
            for k in self.layernames:
                path.append(kwargs.pop(k))
        except KeyError:
            if kwargs:
                raise KeyError("Incorrect address specification.")
        return path

    # Getters

    def __getitem__(self, key):
        """Convenience function equivalent to get."""
        if isinstance(key, tuple):
            return self.get(*key)
        else:
            return self.get(key)

    def _folio(self, *args, **kwargs):
        """Returns a Folio from a path specification.

        This function accepts truncated addresses but does not perform
        random search as subset would.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching path can be found.
        :returns: an NxFolio.
        """
        if isinstance(self.root, fol.NxDocument):
            if args or kwargs:
                raise KeyError
            return self.root
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
                if not isinstance(origin, fol.NxPortFolio):
                    raise KeyError
                tab, *path = path
                if path:
                    copy[tab] = origin[tab].proxy()
                    origin = origin[tab]
                    copy = copy[tab]
                else:
                    copy[tab] = origin[tab].copy()
        except KeyError:
            msg = "No branch matching the path specification."
            raise KeyError(msg)
        return subregistry

    def _args_subset(self, *args):
        """Subset primitive based on *args."""
        if not args:
            return NxSubRegistry(self, copy=True)
        if not isinstance(self.root, fol.NxPortFolio):
            return NxSubRegistry(self, empty=True)
        subregistry = NxSubRegistry(self)
        folios = self.root.search(*args)
        for f in folios:
            subregistry.root.insert(f.copy(), *f.path)
        return subregistry

    def _kwargs_subset(self, **kwargs):
        """Subset primitive based on **kwargs."""
        if not kwargs:
            return NxSubRegistry(self, copy=True)
        if not isinstance(self.root, fol.NxPortFolio):
            return NxSubRegistry(self, empty=True)
        subregistry = NxSubRegistry(self)
        branches = [self.root]
        depth = 0
        if 'key' in self.layernames:
            layernames = self.layernames[:-1]
        else:
            layernames = self.layernames
        for i, k in enumerate(layernames):
            try:
                v = kwargs.pop(k)
            except KeyError:
                continue
            new_branches = []
            for b in branches:
                for f in b.subfolios(i - depth):
                    try:
                        new_branches.append(f[v])
                    except KeyError:
                        pass
            if not new_branches:
                return subregistry
            branches = new_branches
            depth = i + 1
        if kwargs:
            return subregistry
        for f in new_branches:
            subregistry.root.insert(f.copy(), *f.path)
        return subregistry

    @fun.args_or_kwargs
    def subset(self, *args, **kwargs):
        """Returns a subregistry from a random access path search.

        Unlike branch, the arguments can provide partial path specifications
        and even point to multiple, separate branches.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if the path specification uses incorrect keywords.
        :returns: a NxSubRegistry.
        """
        if args:
            return self._args_subset(*args)
        else:
            return self._kwargs_subset(**kwargs)

    def branches(self, *args, **kwargs):
        """Returns iterable of paths to leaf folios.

        This function accepts truncated addresses but does not perform
        random search as subset would.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of paths to foios, as tuples.
        """
        root = self._folio(*args, **kwargs)
        if root.leaf:
            return iter((root.path,))
        return (l.path for l in root.leaves())

    def addresses(self, *args, **kwargs):
        """Returns iterable of registered entry addresses, subset by arguments.

        Technically this function returns addresses to documents,
        complemented by keys in the case of NxSchedules.
        This function accepts truncated addresses but does not perform
        random search as subset would.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of addresses to objects, as tuples.
        """
        root = self._folio(*args, **kwargs)
        if isinstance(root, fol.NxPortFolio):
            leaves = root.leaves()
        else:
            leaves = (root,)
        results = []
        for leaf in leaves:
            path = leaf.path
            if isinstance(leaf, fol.NxSchedule):
                results.extend((path + (k,) for k in leaf.keys()))
            elif isinstance(leaf, fol.NxDocument):
                results.append(path)
        return iter(results)

    def values(self, *args, **kwargs):
        """Retuns an iterable of entries whose address match the arguments.

        This function accepts truncated addresses but does not perform
        random search as subset would.
        Note that this function returns document entries but not portfolio
        titles.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of Registrables.
        """
        return self._folio(*args, **kwargs).read()

    def items(self, *args, **kwargs):
        """Retuns an (address, obj) iterable where address match the arguments.

        This function accepts truncated addresses but does not perform
        random search as subset would.
        Note that this function returns document entries but not portfolio
        titles.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of (address, Registrable) tuples.
        """
        root = self._folio(*args, **kwargs)
        if isinstance(root, fol.NxPortFolio):
            leaves = root.leaves()
        else:
            leaves = (root,)
        results = []
        for leaf in leaves:
            path = leaf.path
            if isinstance(leaf, fol.NxSchedule):
                results.extend(((path + (k,), v) for k, v in leaf.items()))
            elif isinstance(leaf, fol.NxDocument):
                results.extend((path, v) for v in leaf.read())
        return iter(results)

    def scan(self, *args, **kwargs):
        """Returns an iterable of objects whose address supersedes the spec.

        :param args: a partial sequential registry address specification.
        :param kwargs: a partial named registry address specification.
        :returns: a Registrable.
        """
        subregistry = self.subset(*args, **kwargs)
        return subregistry.values()

    def titles(self, *args, **kwargs):
        """Retuns an iterable of titles whose address match the arguments.

        This function accepts truncated addresses but does not perform
        random search as subset would. None titles are ignored.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of Registrables.
        """
        root = self._folio(*args, **kwargs)
        return root.titles(root=True)

    def browse(self, *args, **kwargs):
        """Retuns an iterable of titles whose address match the arguments.

        browse differs from titles in that it allows random address access,
        i.e. partial address specifications.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :returns: an iterable of Registrables.
        """
        subregistry = self.subset(*args, **kwargs)
        return subregistry.titles()

    def get(self, *args, **kwargs):
        """Returns a single title or raises an exception.

        :param args: a sequential registry address specification.
        :param kwargs: a named registry address specification.
        :raises KeyError: if the registry address doesn't exist.
        :returns: an Registrable.
        """
        return self._folio(*args, **kwargs).title

    def fetch(self, *args, **kwargs):
        """Returns a single title or raises an exception.

        fetch differs from get in that it allows random address access, i.e.
        partial address specifications.

        :param args: a partial sequential registry address specification.
        :param kwargs: a partial named registry address specification.
        :raises KeyError: if no matching registry address exists.
        :raises ValueError: if the registry address points to multiple items.
        :returns: a Registrable.
        """
        try:
            return self.get(*args, **kwargs)
        except KeyError:
            pass
        candidates = list(self.browse(*args, **kwargs))
        if not candidates:
            raise KeyError
        elif len(candidates) > 1:
            raise ValueError
        else:
            return candidates.pop()

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
        rtname = type(self.root).__name__
        rgname = type(self).__name__
        return repr(self.root).replace(rtname, rgname)

    def __str__(self):
        return str(self.root)


class NxRegistry(NxRegistryBase):
    """Base class for registries, with general-purpose functionalities.

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
    __root__ = fol.NxPortFolio  # Placeholder for the folio type of the root.
    __layers__ = ()  # Placeholder for the name and type of folio layers.
    __recurse__ = None  # Placeholder for a recursive folio type.

    def __init__(self, *layers, recurse=None):
        """Instances may overrides layers and recurse definitions."""
        self.layers = layers or self.__layers__
        self.recurse = recurse or self.__recurse__
        super().__init__()

    @xprops.cachedproperty
    def layernames(self):
        try:
            bottom_layer_type = self.layertypes[-1]
        except IndexError:
            bottom_layer_type = self.__root__ if not self.recurse else None
        add_key = bottom_layer_type == fol.NxSchedule
        try:
            names = list(list(zip(*self.layers))[0])
        except (TypeError, IndexError):
            names = []
        if add_key:
            names.append('key')
        return tuple(names)

    @xprops.cachedproperty
    def layertypes(self):
        try:
            return tuple(zip(*self.layers))[1]
        except (TypeError, IndexError):
            return tuple(self.layers)

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
            return self.recurse

    def register(self, obj, *args, **kwargs):
        """Open registration method, can be simplified in subclasses.

        :param obj: a Registrable.
        :param *args or **kwargs: an address specification.
        :raises KeyError: if the address specification is incorrect.
        :returns True: if registration is successful.
        """
        err_msg = "Incorrect registration address specification."
        path = self._path(*args, **kwargs)
        folio = self.root
        i = -1  # Layer counter.
        while path:
            if isinstance(folio, fol.NxPortFolio):
                tab, *path = path
                i += 1
                try:
                    folio = folio[tab]
                except KeyError:
                    try:
                        insert = self.layertype(i)()
                        folio[tab] = insert
                        folio = insert
                    # Failure to create insert or incorrect tab type.
                    except (KeyError, TypeError):
                        raise KeyError(err_msg)
            else:
                break
        else:
            try:
                folio.register(obj)
                return True
            except TypeError:
                raise KeyError(err_msg)
        try:
            folio.register(obj, *path)
            return True
        except TypeError:
            raise KeyError(err_msg)

    def unregister(self, obj, *args, **kwargs):
        """Unregisters object if found, otherwise silences exceptions."""
        if isinstance(self.root, fol.NxDocument):
            return self.root.unregister(obj)
        path = self._path(*args, **kwargs)
        try:
            folio = self.root.retrieve(*path)
            return folio.unregister(obj)
        except KeyError:  # Assume an NxSchedule terminal folio.
            folio = self.root.retrieve(*path[:-1])
            return folio.unregister(obj, path[-1])
        except TypeError:  # NxSchedule missing tab arg in unregister
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

# =============================================================================
# Base registry types
# =============================================================================


class Pool(NxRegistry):
    """A simple registry that provides membership functionality only."""
    __root__ = fol.NxPage

    def __init__(self):
        super().__init__()

    def register(self, obj):
        self.root.register(obj)
        return True

    def unregister(self, obj):
        return self.root.unregister(obj)


class Hierarchy(NxRegistry):
    """A set hierarchy of individual objects.

    The Hierarchy is primed with a root PortFolio, and additional
    layers are specified by name only, being automatically set to
    the PortFolio type. This registry works with titles only.
    """
    __root__ = fol.NxPortFolio

    def __init__(self, *layer_names):
        layers = ((n, fol.NxPortFolio) for n in layer_names)
        super().__init__(*layers)


class Tree(NxRegistry):
    """A tree-like registry of individual objects.

    The Tree specifies no set levels, so it can grow recursively
    pseudo-indefinitely.
    """
    __root__ = fol.NxPortFolio
    __recurse__ = fol.NxPortFolio

    def __init__(self):
        super().__init__()

    def register(self, obj, *args, **kwargs):
        path = self._path(*args, **kwargs)
        folio = self.recurse(obj)
        self.root.insert(folio, *path)
        return True

    def unregister(self, obj, *args, **kwargs):
        try:
            folio = self._folio(*args, **kwargs)
            folio.dispose()
            return True
        except KeyError:
            pass


class Book(NxRegistry):
    """A registry built as a sequence of unordered collections (pages)."""
    __root__ = fol.NxVolume
    __layers__ = [('page', fol.NxPage)]

    def __init__(self):
        super().__init__()


class Roll(NxRegistry):
    """A registry that stores sequential entries.

    This is functionally equivalent to a weak sorted set.
    """
    __root__ = fol.NxScroll

    def __init__(self):
        super().__init__()

    def register(self, obj):
        self.root.register(obj)
        return True

    def unregister(self, obj):
        return self.root.unregister(obj)


class Directory(NxRegistry):
    """A registry that stores indexed entries.

    This is functionally equivalent to a weakvalues dictionary.
    """
    __root__ = fol.NxSchedule

    def __init__(self):
        super().__init__()

    def register(self, obj, key):
        self.root.register(obj, key)
        return True

    def unregister(self, obj, key):
        return self.root.unregister(obj, key)

    def get(self, key):
        return self.root[key]

    def fetch(self, key):
        return self.root.get[key]
