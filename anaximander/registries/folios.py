#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements folios i.e. specialized data structures used to build registries.

Folios are divided into two categories: portfolios hold other folios,
whereas documents are necessarily terminal nodes in the folio structure in
which they reside.
PortFolios carry a title attribute to which an nxobject can be assigned.
This is primarily useful for building object hierarchies or genealogies.
Documents don't have titles but carry entries, either a single entry (NxCard)
or a collection (NxPage, NxScroll, NxSchedule). References to titles and
entries are weak, allowing the content of folios to change dynamically and
not hinder the garbage collection of application objects.
Moreover, the references from portfolios to the folios they contain (tabs) are
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
the registry to folio is weak. Removal of the parent (deletion or setting
it to None) will cause removal of the title / entries, which in turn will
throw the folio into the garbage collector.
The second enabler of the design is the _folios attribute of nxobjects,
which a strong set to all the folios that hold a particular nxobject. This
strong reference is what ensures that a document can stay alive in memory
for so long as it contains at least one object, since as a terminal node it
cannot have a child pointing to it.
As a result, portofolios can be strongly referenced by either children or
their title, but if they have neither they get garbage collected.
Folios are designed to create relatively shallow data structures employed
by different types of registries, i.e. caches, indexes, glossaries, etc.
Another layer implementation are the folio proxies, which behave like a
folio and have a parent attribute exactly like NxFolio, but do not hold
direct references to their titles / entries, instead relying on a weakly
referenced folio instance. Proxies are necessary to enable effective
subsetting of registries: registry subsets are themselves registries, but
they contain folio proxies rather than folios, hence avoiding
unnecessary object copies. NxFolio objects have a _proxies attribute that
works in a similar fashion to nxobjects' _folios attribute.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import abc
import weakref
from itertools import chain
import numbers
from weakref import WeakSet, WeakValueDictionary

from blist import sortedset, weaksortedset

from .. import nxobject
from ..utilities import functions as fun
from ..utilities import xprops

__all__ = ['NxFolder', 'NxVolume',
           'NxCard', 'NxPage', 'NxScroll', 'NxSchedule']

#==============================================================================
### Abstract base classes for folios and registries.
#==============================================================================


class NxRegistryABC(nxobject, abc.ABC):
    """ABC for registries, in the registries module."""
    pass


class NxFolio(nxobject, abc.ABC):
    """ABC for all folio objects."""
    __print_spacer__ = ' ' * 2

    def __init__(self, **kwargs):
        self._parent = None
        self._path = ()

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
        if isinstance(val, NxRegistryABC):
            self._parent = weakref.ref(val)
        elif isinstance(val, NxPortFolio):
            self._parent = val
        elif val is None:
            self.dispose()
            self._parent = None
            self._path = ()
        else:
            msg = "The Parent to a folio can only be an NxRegistry or an " + \
                  "NxPortFolio."
            raise TypeError(msg)

    @parent.deleter
    def parent(self):
        """Deleting the parent forces a cleanup of the folio."""
        self.dispose()
        self._parent = None
        self._path = ()

    @abc.abstractproperty
    def leaf(self):
        """True if self doesn't have hierarchical children."""
        pass

    @property
    def path(self):
        return self._path

    @property
    def tab(self):
        """The key with which the folio is accessed by its parent."""
        if self._path:
            return self._path[-1]
        return None

    @abc.abstractmethod
    def read(self):
        """Returns an iterable of document entries."""
        pass

    @abc.abstractmethod
    def dispose(self):
        """Disposes of self's content (subfolios, title, entries)."""
        pass

    @abc.abstractmethod
    def register(self, obj, *args, **kwargs):
        """Registers an nxobject, implementation specific to each class."""
        pass

    @abc.abstractmethod
    def unregister(self, obj, *args, **kwargs):
        """Unregisters an nxobject, implementation specific to each class."""
        pass

    def copy(self):
        """Returns a copy of self using proxies."""
        return self.__copy__()

    def deepcopy(self):
        """Returns a copy of self using new folio instances."""
        return self.__deepcopy__()

    def hardcopy(self):
        """Returns a copy of self using strong-reference data structures."""
        return self.__hardcopy__()

    @abc.abstractmethod
    def __copy__(self):
        """Primitive to copy."""
        pass

    @abc.abstractmethod
    def __deepcopy__(self):
        """Primitive to deepcopy."""
        pass

    @abc.abstractmethod
    def __hardcopy__(self):
        """Primitive to hardcopy."""
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

    @xprops.cachedproperty
    def _proxies(self):
        """A set of strong references to self's proxies."""
        return set()

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

    def keys(self):
        superkeys = super().keys()
        try:
            return iter(sorted(superkeys))
        except TypeError:  # Unorderable keys.
            return iter(superkeys)

    def items(self):
        superitems = super().items()
        try:
            return iter(sorted(superitems, key=lambda i: i[0]))
        except TypeError:
            return iter(superitems)

    def values(self):
        try:
            return iter(list(zip(*self.items()))[1])
        except IndexError:
            return iter([])

    def portfolios(self):
        """Returns values, restricted to portfolios."""
        return (f for f in self.values() if isinstance(f, NxPortFolio))

    def retrieve(self, *path):
        """Retrieves a folio at path or raises a KeyError."""
        folio = self
        while path:
            tab, *path = path
            if isinstance(folio, NxDocument):
                raise KeyError
            folio = folio[tab]
        return folio

    def search(self, *path):
        """Returns an iterable of folios whose path is a superset of path.

        :param path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: an iterable of folios.
        """
        if not path:
            results = [self]
        else:
            tab, *path = path
            results = []
            for k, f in self.items():
                if k == tab:
                    if isinstance(f, NxPortFolio):
                        results.extend(f.search(*path))
                    elif not path:
                        results.append(f)
                elif isinstance(f, NxPortFolio):
                    results.extend(f.search(tab, *path))
        return iter(results)

    def find(self, *path):
        """Returns a folio whose path is a superset of path.

        If none is found, a KeyError is raised. If the search path returns
        multiple results, a ValueError is raised.

        :param path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :raises KeyError: if the search path yields no result.
        :raises ValueError: if the seach path yields multiple results.
        :returns: a folio.
        """
        folios = list(self.search(*path))
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
        By convention, depth 0 adds self to the return results.

        :param *depths: optional sequence of integers representing tree depth.
        :returns: an iterable of folios.
        """
        depths = sorted(depths)
        if 0 in depths:
            results = [self]
            depths.pop(0)
            if not depths:
                return iter(results)
        else:
            results = []
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

    def titles(self, *depths, root=False):
        """Returns an iterable of non-None titles in self's subfolios.

        :param *depths: optional sequence of integers representing tree depth.
        :param root: if True, self's title is included in the results.
        :returns: an iterable of titles.
        """
        results = []
        if root and self.title is not None:
            results.append(self.title)
        for f in self.subfolios(*depths):
            if isinstance(f, NxPortFolio) and f.title is not None:
                results.append(f.title)
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

    def read(self):
        """Returns an iterable of document entries."""
        return chain(*(l.read() for l in self.leaves()))

    # Setters

    def __setitem__(self, tab, folio):
        if tab in self:
            self[tab].dispose()
        if isinstance(folio, NxFolio):
            folio.parent = self
            folio._path = self.path + (tab,)
        else:
            msg = "Only NxFolio objects can be inserted into an NxPortFolio."
            raise TypeError(msg)
        super().__setitem__(tab, folio)

    def insert(self, folio, *path):
        """Inserts folio at specified path, creating portfolios as needed."""
        try:
            tab, *path = path
        except ValueError:
            raise TypeError("Insert requires at least one tab.")
        if path:
            try:
                self[tab].insert(folio, *path)
            except KeyError:
                self.sub(tab).insert(folio, *path)
            except AttributeError:
                msg = "Path points inside an NxDocument."
                raise KeyError(msg)
        else:
            self[tab] = folio

    def __delitem__(self, tab):
        folio = super().pop(tab)  # May raise a KeyError
        del folio.parent

    def clear(self):
        """Cleanly disassembles self's substructure."""
        folios = list(self.values())
        for f in folios:
            del f.parent

    def dispose(self):
        """Disposes of self by clearing its subfolios and title."""
        self.clear()
        del self.title

    def sub(self, *tab, title=None, type_=None):
        """Creates a subfolio of self.

        Key is generally required but subclasses can implement rules
        to make it optional.

        :param tab: An optional tab specification to be inserted in self.
        :param title: An optional title to the subfolio.
        :type_: The type of the subfolio, defaulting to type(self).
        :returns: A folio, after it's been inserted.
        """
        type_ = type_ or type(self)
        sub = type_(title=title)
        self[tab[0]] = sub
        return sub

    def pop(self, tab, *args):
        return NotImplemented

    def popitem(self):
        return NotImplemented

    def setdefault(self, tab, default=None):
        return NotImplemented

    def update(self, dict=None, **kwargs):
        return NotImplemented

    @classmethod
    def fromkeys(cls, *args, **kwargs):
        return NotImplemented

    # Registration methods

    def register(self, obj):
        self.title = obj

    def unregister(self, obj):
        if self.title == obj:
            del self.title
            return True
        return False

    # Administrative functions

    def copy(self):
        """Returns a copy of self using proxies."""
        return self.__copy__()

    def deepcopy(self):
        """Returns a copy of self using new folio instances."""
        return self.__deepcopy__()

    def hardcopy(self):
        """Returns a copy of self using strong-reference data structures."""
        return self.__hardcopy__()

    def proxy(self):
        """Returns a proxy to self, empty of subfolios."""
        type_ = globals()[type(self).__name__ + 'Proxy']
        return type_(self)

    def __copy__(self):
        """Primitive to copy."""
        copy_ = self.proxy()
        for k, f in self.items():
                copy_[k] = f.__copy__()
        return copy_

    def __deepcopy__(self):
        """Primitive to deepcopy."""
        copy_ = type(self)(title=self.title)
        for k, f in self.items():
                copy_[k] = f.__deepcopy__()
        return copy_

    class TitledDict(dict):
        """A dictionary with a title attribute."""

        @xprops.settablecachedproperty
        def title(self):
            return None

    def __hardcopy__(self):
        """Primitive to hardcopy."""
        copy_ = self.TitledDict()
        copy_.title = self.title
        for k, f in self.items():
            copy_[k] = f.__hardcopy__()
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

    def __setitem__(self, tab, folio):
        if not isinstance(tab, str):
            raise TypeError
        super().__setitem__(tab, folio)


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

    def __setitem__(self, tab, folio):
        if not isinstance(tab, numbers.Integral):
            raise TypeError
        if not tab >= 1:
            raise ValueError
        super().__setitem__(tab, folio)

#==============================================================================
### Document classes
#==============================================================================


class NxDocumentBase(NxFolio):
    """Base class for NxDocument and NxDocumentProxy."""

    @property
    def leaf(self):
        return True


class NxDocument(NxDocumentBase):
    """ABC for terminal folios that contain entries."""

    @xprops.cachedproperty
    def _proxies(self):
        """A set of strong references to self's proxies."""
        return set()

    @abc.abstractmethod
    def read(self):
        """Returns a stable iterable of entries in self."""
        pass

    def copy(self):
        """Returns a copy of self using proxies."""
        return self.__copy__()

    def deepcopy(self):
        """Returns a copy of self using new folio instances."""
        return self.__deepcopy__()

    def hardcopy(self):
        """Returns a copy of self using strong-reference data structures."""
        return self.__hardcopy__()

    def proxy(self):
        """Returns a proxy to self."""
        type_ = globals()[type(self).__name__ + 'Proxy']
        return type_(self)

    __copy__ = proxy

    def _str(self, *indent):
        """Primitive to __str__."""
        return type(self).__name__[2:]

    def __str__(self):
        return self._str()


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

    def read(self):
        """Returns a stable iterable of entries in self."""
        return iter([self.entry])

    def dispose(self):
        del self.entry

    def register(self, obj):
        """Makes obj the entry."""
        self.entry = obj

    def unregister(self, obj):
        """If obj is the entry, it gets erased."""
        if obj == self.entry:
            del self.entry
            return True
        return False

    def __deepcopy__(self):
        """Primitive to deepcopy."""
        return type(self)(self.entry)

    def __hardcopy__(self):
        """Primitive to hardcopy."""
        return self.entry


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

    def copy(self):
        """Returns a copy of self using proxies."""
        return self.__copy__()

    def deepcopy(self):
        """Returns a copy of self using new folio instances."""
        return self.__deepcopy__()

    def hardcopy(self):
        """Returns a copy of self using strong-reference data structures."""
        return self.__hardcopy__()

    def __repr__(self):
        cls = type(self).__name__
        if not self:
            return '{cls}()'.format(**locals())
        content = fun.spformat(self)
        return '{cls}({content})'.format(**locals())

    def _str(self, *indent):
        """Primitive to __str__."""
        return type(self).__name__[2:] + ' | ' + fun.spformat(self)

    def __str__(self):
        return self._str()


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

    def read(self):
        """Returns a stable iterable of entries in self."""
        return iter(list(self))

    def dispose(self):
        self.clear()

    def register(self, obj):
        """Adds obj to the document."""
        self.add(obj)

    def unregister(self, obj):
        """If obj is on the document, it gets erased."""
        try:
            self.remove(obj)
            return True
        except KeyError:
            return False

    def __deepcopy__(self):
        """Primitive to deepcopy."""
        copy_ = type(self)()
        for item in self:
            copy_.add(item)
        return copy_

    def __hardcopy__(self):
        """Primitive to hardcopy."""
        return set(self)

fun.monkeypatch(NxPage, _NxEntrySetMixin)


class NxScroll(weaksortedset, NxDocument):
    """A document that keeps a sorted set of entries."""
    __key__ = None  # class variable for a sort key.

    def __init__(self, entries=None, *, key=None, **kwargs):
        super().__init__(key=fun.get(key, self.__key__))
        NxFolio.__init__(self)
        if entries is not None:
            for e in entries:
                self.add(e)

    @property
    def key(self):
        return self._key

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

    def read(self):
        """Returns a stable iterable of entries in self."""
        return iter(list(self))

    def dispose(self):
        self.clear()

    def register(self, obj):
        """Adds obj to the document."""
        self.add(obj)

    def unregister(self, obj):
        """If obj is on the document, it gets erased."""
        try:
            self.remove(obj)
            return True
        except KeyError:
            return False

    def __deepcopy__(self):
        """Primitive to deepcopy."""
        copy_ = type(self)(key=self.key)
        for item in self:
            copy_.add(item)
        return copy_

    def __hardcopy__(self):
        """Primitive to hardcopy."""
        return sortedset(self, key=self.key)

fun.monkeypatch(NxScroll, _NxEntrySetMixin)


class NxSchedule(WeakValueDictionary, NxDocument):
    """An indexed entry container."""

    def __init__(self, entries=None, **kwargs):
        super().__init__()
        NxFolio.__init__(self)
        if entries is not None:
            for k, v in entries:
                self[k] = v
        for k, v in kwargs:
            self[k] = v

    # Hashability

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __ne__(self, other):
        return id(self) != id(other)

    # Getters

    def read(self):
        """Returns a stable iterable of self's objects."""
        return iter(list(self.values()))

    # Setters

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        try:
            value._folios.add(self)
        except AttributeError:
            msg = "Only nxobjects can be entries to an NxDocument."
            raise TypeError(msg)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        item = super().pop(key)  # May raise a KeyError
        item._folios.remove(self)

    def clear(self):
        for key in list(self):
            del self[key]

    def dispose(self):
        self.clear()

    def pop(self, key, *args):
        return NotImplemented

    def popitem(self):
        return NotImplemented

    def setdefault(self, key, default=None):
        return NotImplemented

    def update(self, dict=None, **kwargs):
        return NotImplemented

    @classmethod
    def fromkeys(cls, *args, **kwargs):
        return NotImplemented

    def register(self, obj, key):
        """Adds obj to the document."""
        self[key] = obj

    def unregister(self, obj, key):
        """If obj is on the document, it gets erased."""
        try:
            if self[key] == obj:
                del self[key]
                return True
        except KeyError:
            pass
        return False

    # Administrative functions

    def copy(self):
        """Returns a copy of self using proxies."""
        return self.__copy__()

    def deepcopy(self):
        """Returns a copy of self using new folio instances."""
        return self.__deepcopy__()

    def hardcopy(self):
        """Returns a copy of self using strong-reference data structures."""
        return self.__hardcopy__()

    def __copy__(self):
        """Primitive to copy."""
        type_ = globals()[type(self).__name__ + 'Proxy']
        return type_(self)

    def __deepcopy__(self):
        """Primitive to deepcopy."""
        copy_ = type(self)()
        for k, v in self.items():
                copy_[k] = v
        return copy_

    def __hardcopy__(self):
        """Primitive to hardcopy."""
        return dict(self)

    def __reduce__(self):
        return NotImplemented

    def __repr__(self):
        cls = type(self).__name__
        if not self:
            return '{cls}()'.format(**locals())
        content = fun.spformat(self)
        return '{cls}({content})'.format(**locals())

    def _str(self, *indent):
        """Primitive to __str__."""
        return type(self).__name__[2:] + ' | ' + fun.spformat(self)

#==============================================================================
### Folio proxies
#==============================================================================


class NxFolioProxy(NxFolio):
    """Base class for folio proxies, lightweight references to folios."""

    def __init__(self, folio):
        super().__init__()
        self._folio = weakref.ref(folio)
        folio._proxies.add(self)

    @property
    def folio(self):
        try:
            return self._folio()
        except AttributeError:
            return None

    def dispose(self):
        self.folio._proxies.remove(self)
        del self._folio

    def register(self, obj, *args, **kwargs):
        self.folio.register(obj, *args, **kwargs)

    def unregister(self, obj, *args, **kwargs):
        return self.folio.unregister(obj, *args, **kwargs)

    def __copy__(self):
        return type(self)(self.folio)


class NxPortFolioProxy(NxFolioProxy, NxPortFolio):
    """A lightweight, read-only weak reference to an NxPortFolio object."""

    @property
    def title(self):
        return self.folio.title

    def insert(self, proxy, *path):
        """Inserts proxy at specified path, creating portfolios as needed."""
        try:
            tab, *path = path
        except ValueError:
            raise TypeError("Insert requires at least one tab.")
        if path:
            try:
                self[tab].insert(proxy, *path)
            except KeyError:
                try:
                    self[tab] = self.folio[tab].proxy()
                except KeyError:
                    msg = "Cannot insert into a proxy if tab does not " + \
                          "exist in the referred folio."
                    raise KeyError(msg)
                try:
                    self[tab].insert(proxy, *path)
                except AttributeError:
                    msg = "Path points inside an NxDocument."
                    raise KeyError(msg)
            except AttributeError:
                msg = "Path points inside an NxDocument."
                raise KeyError(msg)
        else:
            self[tab] = proxy

    def dispose(self):
        self.clear()
        super().dispose()

    def __copy__(self):
        copy_ = type(self)(self.folio)
        for k, f in self.items():
                copy_[k] = f.__copy__()
        return copy_

    def __deepcopy__(self):
        type_ = globals()[type(self).__name__[:-5]]
        copy_ = type_(self.title)
        for k, f in self.items():
            copy_[k] = f.__deepcopy__()
        return copy_

    def __hardcopy__(self):
        copy_ = NxPortFolio.TitledDict()
        copy_.title = self.title
        for k, f in self.items():
            copy_[k] = f.__hardcopy__()
        return copy_


class NxFolderProxy(NxPortFolioProxy, NxFolder):
    """Proxy to an NxFolder."""
    pass


class NxVolumeProxy(NxPortFolioProxy, NxVolume):
    """Proxy to an NxVolume."""
    pass


class NxDocumentProxy(NxFolioProxy):
    """A lightweight, read-only weak reference to an NxDocument object."""

    @property
    def leaf(self):
        return True

    def read(self):
        return self.folio.read()

    def _str(self, *indent):
        return self.folio._str()

    def __str__(self):
        return self.folio.__str__()

    def __repr__(self):
        frepr = self.folio.__repr__()
        ftype = type(self.folio).__name__
        ptype = type(self).__name__
        return frepr.replace(ftype, ptype)


class NxCardProxy(NxDocumentProxy):
    """A proxy to an an NxCard."""

    @property
    def entry(self):
        return self.folio.entry

    def __deepcopy__(self):
        return NxCard(self.entry)

    def __hardcopy__(self):
        return self.entry


class _CollectionDocumentProxy(NxDocumentProxy):
    """Parent class to NxPage, NxScroll, NxSchedule proxies."""

    def __getitem__(self, key):
        return self.folio.__getitem__(key)


class NxPageProxy(_CollectionDocumentProxy):
    """Proxy to an NxPage."""

    def __deepcopy__(self):
        copy_ = NxPage()
        for obj in self.read():
            copy_.add(obj)
        return copy_

    def __hardcopy__(self):
        return set(self.folio)


class NxScrollProxy(_CollectionDocumentProxy):
    """Proxy to an NxScroll."""

    @property
    def key(self):
        return self.folio.key

    def __deepcopy__(self):
        copy_ = NxScroll(key=self.key)
        for obj in self.read():
            copy_.add(obj)
        return copy_

    def __hardcopy__(self):
        return sortedset(self.folio, key=self.key)


class NxScheduleProxy(_CollectionDocumentProxy):
    """Proxy to an NxSchedule."""

    def keys(self):
        return self.folio.keys()

    def values(self):
        return self.folio.values()

    def items(self):
        return self.folio.items()

    def __deepcopy__(self):
        copy_ = NxSchedule()
        for key, obj in self.folio.items():
            copy_[key] = obj
        return copy_

    def __hardcopy__(self):
        return dict(self.folio)
