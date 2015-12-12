#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements folios i.e. specialized data structures used to build registries.

Folios are divided into two categories: portfolios hold other folios,
whereas documents are necessarily terminal nodes in the folio structure in
which they reside.
Portfolios carry a title attribute to which an nxobject can be assigned.
Documents carry entries, either a single entry (NxCard) or a collection of
entries (NxPage, NxScroll). References to titles and entries are weak,
allowing the content of folios to change dynamically and not hinder
the garbage collection of application objects.
Moreover, the references from portfolios to the folios they contain are
also weak references. This enables folio structures themselves to be dynamic
as well. For instance, a page of an index may be garbage collected once
it holds no objects.
To enable this design and make hierchical structures of portfolios hold
together, each folio has a parent property that either points to a
portofolio that holds it, or to a registry object. A registry object is
an interface that allows registration / unregistration into a folio structure.
If the parent of a folio is a portfolio, then the reference is strong -whereas
the reference from the portfolio to the folio is weak. If the parent of a
folio is a registry, then the reference is weak, whereas the reference from
the registry to folio is weak. Folio types *must* have a parent. Instantiation
without a parent will fail, and removal of the parent (deletion or setting
it to None) will cause removal of the title / entries, which in turn will
throw the folio into the garbage collector.
The second enabler of the design is the _nxfolios attribute of nxobjects,
which a strong set to all the folios that hold a particular nxobject. This
strong reference is what ensures that a document can stay alive in memory
for so long as it contains at least one object, since as a terminal node it
cannot have a child pointing to it.
As a result, portofolios can be strongly referenced by either children or
their title, but if they have neither they get garbage collected.
Folios are designed to create relatively shallow data structures employed
by different types of registries, i.e. caches, indexes, glossaries, etc.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import abc
import weakref
from weakref import WeakSet, WeakValueDictionary

from blist import weaksortedset

from ..arche import nxobject
from ..utilities import functions as fun
from ..utilities import xprops

#==============================================================================
### Abstract base classes for folios and registries.
#==============================================================================


class NxRegistry(nxobject, abc.ABC):
    """ABC for registries, in the registries module."""

    @abc.abstractmethod
    def register(self, obj, *address):
        pass

    @abc.abstractmethod
    def unregister(self, obj, *address):
        pass


class NxFolio(nxobject, abc.ABC):
    """ABC for all folio objects."""
    __print_spacer__ = ' ' * 2

    def __init__(self, **kwargs):
        self._parent = None

    @property
    def parent(self):
        """Returns the portfolio or registry parent."""
        val = self._parent
        if isinstance(val, weakref.ref):
            return val()
        return val

    @parent.setter
    def parent(self, val):
        """Sets the parent, which must be portofolio or registry or None.

        If the parent is set to None, the folio is disposed of. Specifics of
        the dispose method are particular to each subclass, but systematically
        involves clearing subfolios, title and entries -depending on the
        nature of the folio.

        :param val: an NxRegistry or NxPortfolio instance or None.
        :raises TypeError: if val is of the wrong type.
        """
        if isinstance(val, NxRegistry):
            self._parent = weakref.ref(val)
        elif isinstance(val, NxPortFolio):
            self._parent = val
        elif val is None:
            self.dispose()
            self._parent = None
        else:
            msg = "The Parent to a folio can only be an NxRegistry or an " + \
                  "NxPortFolio."
            raise TypeError(msg)

    @parent.deleter
    def parent(self):
        """Deleting the parent forces a cleanup of the folio."""
        self.dispose()
        self._parent = None

    @abc.abstractproperty
    def leaf(self):
        """True if self doesn't have hierarchical children."""
        pass

    @abc.abstractmethod
    def dispose(self):
        """Disposes of self's content (subfolios, title, entries)."""
        pass

    def copy(self):
        """Returns a deep copy of self."""
        return self.__copy__()

    @abc.abstractmethod
    def __copy__(self):
        """Primitive to copy."""
        pass

    @abc.abstractmethod
    def _str(self, *indent):
        """Primitive to __str__."""
        pass

#==============================================================================
### Portfolio classes
#==============================================================================


class NxPortFolio(WeakValueDictionary, NxFolio, abc.ABC):
    """Base class for folios that carry other folios.

    Portfolios have an optional title attribute that must be an nxobject.
    The title is weakly held, but it itself holds a strong reference to
    the portfolio.
    """

    def __init__(self, title=None, **kwargs):
        super().__init__()
        NxFolio.__init__(self)
        if title is not None:
            self.title = title

    # Hashability

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __ne__(self, other):
        return id(self) != id(other)

    # Properties

    @xprops.weakproperty
    def title(self):
        return None

    @title.setter
    def title(self, val=None):
        if self.title is not None:
            self.title._folios.remove(self)
            del self._title
        if val is not None:
            try:
                val._folios.add(self)
            except AttributeError:
                msg = "Only nxobjects can be titles to an NxPortFolio."
                raise TypeError(msg)
            self._title = weakref.ref(val)

    @title.deleter
    def title(self):
        self.title = None

    @property
    def leaf(self):
        return len(self) == 0

    @property
    def height(self):
        """Returns the maximum depth of self, starting at zero."""
        if not self:
            return 0
        try:
            return max(f.height for f in self.portfolios()) + 1
        except ValueError:
            return 1

    @property
    def subcount(self):
        """Returns the total number of subfolios."""
        return len(list(self.subfolios()))

    # Accessors

    def portfolios(self):
        """Returns values, restricted to portfolios."""
        return (f for f in self.values() if isinstance(f, NxPortFolio))

    def get(self, *path):
        """Retrieves a folio at path or raises a KeyError."""
        folio = self
        while path:
            key, *path = path
            folio = folio[key]
        return folio

    def find(self, *path):
        """Returns an iterable of folios whose path is a superset of path.

        :param path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: an iterable of folios.
        """
        if not path:
            results = [self]
        else:
            key, *path = path
            results = []
            for k, f in self.items():
                if k == key:
                    if isinstance(f, NxPortFolio):
                        results.extend(f.find(*path))
                    elif not path:
                        results.append(f)
                elif isinstance(f, NxPortFolio):
                    results.extend(f.find(key, *path))
        return iter(results)

    def fetch(self, *path):
        """Returns a folio whose path is a superset of path.

        If none is found, a KeyError is raised. If the search path returns
        multiple results, a ValueError is raised.

        :param path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :raises KeyError: if the search path yields no result.
        :raises ValueError: if the seach path yields multiple results.
        :returns: a folio.
        """
        folios = list(self.find(*path))
        if not folios:
            raise KeyError
        elif len(folios) == 1:
            return folios[0]
        else:
            raise ValueError

    def subfolios(self, *depths):
        """Returns an iterable of subfolios at arbitrary depths.

        This function recursively returns folios that descend from self,
        optionally restricted to specific depths. For instance:
        self.subfolios(1) is equivalent to self.values()
        self.subfolios(1, 2) returns descendants at levels 1 and 2.

        :param *depths: optional sequence of integers representing tree depth.
        :returns: an iterable of folios.
        """
        results = []
        depths = sorted(depths)
        if not depths:
            results.extend(self.values())
            for f in self.portfolios():
                results.extend(f.subfolios())
            return iter(results)
        if depths[0] == 1:
            results.extend(self.values())
            depths = depths[1:]
            if not depths:
                return iter(results)
        depths = (d - 1 for d in depths)
        for k, f in self.items():
            results.extend(f.subfolios(*depths))
        return iter(results)

    def leaves(self):
        """Returns an iterable of bottom-level folios."""
        results = []
        for k, f in self.items():
            if f.leaf:
                results.append(f)
            else:
                results.extend(f.leaves())
        return iter(results)

    # Setters

    def __setitem__(self, key, folio):
        if key in self:
            self[key].dispose()
        if isinstance(folio, NxFolio):
            folio.parent = self
        else:
            msg = "Only NxFolio objects can be inserted into an NxPortFolio."
            raise TypeError(msg)
        super().__setitem__(key, folio)

    def insert(self, folio, *path):
        """Inserts folio at specified path, creating portfolios as needed."""
        try:
            key, *path = path
        except ValueError:
            raise TypeError("Insert requires at least one key.")
        if path:
            try:
                self[key].insert(folio, *path)
            except KeyError:
                self.sub(key).insert(folio, *path)
        else:
            self[key] = folio

    def __delitem__(self, key):
        folio = super().pop(key)  # May raise a KeyError
        del folio.parent

    def clear(self):
        """Cleanly disassembles self's substructure."""
        folios = self.values()
        for f in folios:
            del f.parent

    def dispose(self):
        """Disposes of self by clearing its subfolios and title."""
        self.clear()
        del self.title

    def sub(self, *key, title=None, type_=None):
        """Creates a subfolio of self.

        Key is generally required but subclasses can implement rules
        to make it optional.

        :param key: An optional key specification to be inserted in self.
        :param title: An optional title to the subfolio.
        :type_: The type of the subfolio, defaulting to type(self).
        :returns: A folio, after it's been inserted.
        """
        type_ = type_ or type(self)
        sub = type_(title=title)
        self[key[0]] = sub
        return sub

    def pop(self, key, *args):
        return NotImplemented

    def popitem(self):
        return NotImplemented

    def setdefault(self, key, default=None):
        return NotImplemented

    def update(self, dict=None, **kwargs):
        return NotImplemented

    # Administrative functions

    def __copy__(self):
        """Primitive to copy."""
        copy_ = type(self)(title=self.title)
        for k, f in self.items():
                copy_[k] = f.__copy__()
        return copy_

    def __reduce__(self):
        return NotImplemented

    def __repr__(self):
        cls = type(self).__name__
        if not self:
            return '{cls}()'.format(**locals())
        items = fun.spformat(list(self.items()))
        height = 'height={}'.format(self.height)
        subs = fun.spformat(self.subcount)
        return '{cls}({items}, {height}, {subs})'.format(**locals())

    def _str(self, indent=0):
        """Primitive printing function."""
        just = (indent + 1) * self.__print_spacer__
        # max key length:
        try:
            mkl = max((len(str(k))) for k in self.keys())
        except ValueError:
            mkl = 4

        def itemstr(k, v):
            return just + '{:<{}}: {}'.format(k, mkl, v._str(indent + 1))

        header = str(self.title) if self.title else type(self).__name__[2:]
        content = '\n'.join(itemstr(*i) for i in self.items())
        return '\n'.join((header, content)) if content else header

    def __str__(self):
        """A tree-like representation."""
        return self._str()


class NxFolder(NxPortFolio):
    """An indexed Portfolio whose keys are strings."""

    def __setitem__(self, key, folio):
        if not isinstance(key, str):
            raise TypeError
        super().__setitem__(key, folio)

    def keys(self):
        return iter(sorted(super().keys()))

    def items(self):
        return iter(sorted(super().items(), key=lambda i: i[0]))

    def values(self):
        try:
            return iter(list(zip(*self.items()))[1])
        except IndexError:
            return iter([])


class NxVolume(NxPortFolio):
    """A portfolio that provides numerically indexed access to its items.

    Unlike a Python list, the indexing numbers on a volume start at 1.
    """

    def insert(self, folio, *path):
        """Inserts folio at specified path, creating portfolios as needed."""
        if not path:
            try:
                path = (max(self.keys()) + 1,)
            except ValueError:
                path = (1,)
        super().insert(folio, *path)

    def sub(self, *index, title=None, type_=None):
        """Creates a subfolio of self.

        :param index: An optional insertion index.
        :param title: An optional title to the subfolio.
        :type_: The type of the subfolio, defaulting to type(self).
        :returns: A folio, after it's been inserted.
        """
        if not index:
            try:
                index = (max(self.keys()) + 1,)
            except ValueError:
                return (1,)
        return super()(self, *index, title=title, type_=type_)

    def __setitem__(self, key, folio):
        if not isinstance(key, int):
            raise TypeError
        if not key >= 1:
            raise ValueError
        super().__setitem__(key, folio)

    def keys(self):
        return iter(sorted(super().keys()))

    def items(self):
        return iter(sorted(super().items(), key=lambda i: i[0]))

    def values(self):
        try:
            return iter(list(zip(*self.items()))[1])
        except IndexError:
            return iter([])

#==============================================================================
### Document classes
#==============================================================================


class NxDocument(NxFolio, abc.ABC):
    """ABC for terminal folios that contain entries."""

    @property
    def leaf(self):
        return True

    def _str(self, *indent):
        """Primitive to __str__."""
        return type(self).__name__[2:]

    @abc.abstractmethod
    def entries(self):
        """Returns a stable iterable of entries in self."""


class NxCard(NxDocument):
    """A document that holds a single entry."""

    def __init__(self, entry=None, **kwargs):
        NxFolio.__init__(self)
        if entry is not None:
            self.entry = entry

    @xprops.weakproperty
    def entry(self):
        return None

    @entry.setter
    def entry(self, val=None):
        if self.entry is not None:
            self.entry._folios.remove(self)
            del self._entry
        if val is not None:
            try:
                val._folios.add(self)
            except AttributeError:
                msg = "Only nxobjects can be entries to an NxDocument."
                raise TypeError(msg)
            self._entry = weakref.ref(val)

    @entry.deleter
    def entry(self):
        self.entry = None

    def entries(self):
        """Returns a stable iterable of entries in self."""
        return iter([self.entry])

    def dispose(self):
        del self.entry

    def __copy__(self):
        """Primitive to copy."""
        return type(self)(self.entry)


class _NxEntrySetMixin(object):
    """A Mixin class for NxPage and NxScroll."""

    # Hashability

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __ne__(self, other):
        return id(self) != id(other)

    # Setters

    def pop(self):
        return NotImplemented

    def update(self, iterable):
        return NotImplemented

    def intersection_update(self, other):
        return NotImplemented

    def difference_update(self, other):
        return NotImplemented

    def symmetric_difference_update(self, other):
        return NotImplemented

    def clear(self):
        for item in list(self):
            self.remove(item)

    # Administrative functions

    def __repr__(self):
        cls = type(self).__name__
        if not self:
            return '{cls}()'.format(**locals())
        content = fun.spformat(self)
        return '{cls}({content})'.format(**locals())

    def _str(self, *indent):
        """Primitive to __str__."""
        return type(self).__name__[2:] + ' | {}'.format(len(self))


class NxPage(WeakSet, NxDocument):
    """An entry container."""

    def __init__(self, entries=None, **kwargs):
        super().__init__()
        NxFolio.__init__(self)
        if entries is not None:
            for e in entries:
                self.add(e)

    def add(self, item):
        """Overwrites the default add method."""
        try:
            item._folios.add(self)
        except AttributeError:
            msg = "Only nxobjects can be entries to an NxDocument."
            raise TypeError(msg)
        super().add(item)

    def remove(self, item):
        super().remove(item)
        item._folios.remove(self)

    def discard(self, item):
        try:
            self.remove(item)
        except KeyError:
            pass

    def entries(self):
        """Returns a stable iterable of entries in self."""
        return iter(list(self))

    def dispose(self):
        self.clear()

    def __copy__(self):
        """Primitive to copy."""
        copy_ = type(self)()
        for item in self:
                copy_.add(item)
        return copy_

fun.ducktype(NxPage, _NxEntrySetMixin)


class NxScroll(weaksortedset, NxDocument):
    """A document that keeps a sorted set of entries."""
    __key__ = None  # class variable for a sort key.

    def __init__(self, entries=None, *, key=None, **kwargs):
        super().__init__(key=fun.get(key, self.__key__))
        NxFolio.__init__(self)
        if entries is not None:
            for e in entries:
                self.add(e)

    def __getitem__(self, index):
        """A slice returns a weaksortedset rather than an NxScroll."""
        if isinstance(index, slice):
            rv = weaksortedset()
            rv._blist = self._blist[index]
            rv._key = self._key
            return rv
        return super().__getitem__(index)

    def add(self, item):
        """Overwrites the default add method."""
        try:
            item._folios.add(self)
        except AttributeError:
            msg = "Only nxobjects can be entries to an NxDocument."
            raise TypeError(msg)
        super().add(item)

    def remove(self, item):
        super().remove(item)

    def discard(self, item):
        if item in self:
            item._folios.remove(self)
            super().discard(item)

    def entries(self):
        """Returns a stable iterable of entries in self."""
        return iter(list(self))

    def dispose(self):
        self.clear()

    def __copy__(self):
        """Primitive to copy."""
        copy_ = type(self)()
        for item in self:
                copy_.add(item)
        return copy_

fun.ducktype(NxScroll, _NxEntrySetMixin)
