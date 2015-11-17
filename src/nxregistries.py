#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements Anaximander registries, i.e. weak object containers.

The main two registry types are NxCell (a weak set) and NxTree (a recursive
weak values dictionary). The registries are used for building taxonomies,
keeping configuration information, and as instance caches or indexes.
Every NxObject has an attribute _nxregistries which keeps a set of
strong references to the registries in which it is featured.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
# Import statements
#==============================================================================

import abc
from weakref import WeakSet, WeakValueDictionary

from blist import weaksortedset


import xprops

#==============================================================================
# NxWeakContainer base class
#==============================================================================

class NxRegistry(abc.ABC):
    """Abstract base class for NxCell and NxTree.

    The base class implements the parent property, a strong reference
    to a possible parent container of the object, as is the case with
    a subtree or a cell that is a leaf node of a tree.
    NxRegistries can only have one parent and this has implications: the same
    NxRegistry cannot be a node to multiple NxTrees. If something along
    those lines is required, then the structure needs to be copied.
    NxObjects can be featured in multiple NxRegistries, and for that purpose
    they carry a _nxregistries set attribute, which NxRegistry can access and
    alter.
    """

    @xprops.cachedproperty
    def parent(self):
        return None

#    @abc.abstractmethod
#    def register(self, obj, *key):
#        """Registers obj with an optional key, which can be composite."""
#        pass

#==============================================================================
# NxCell class
#==============================================================================

class NxCell(WeakSet, NxRegistry):
    """A weak container of NxObjects."""

    def __init__(self):
        super().__init__()

    def register(self, obj, *key):
        self.add(obj)

#==============================================================================
# NxTree class
#==============================================================================

class NxTree(WeakValueDictionary, NxRegistry):
    """A recursive nested dictionary with weak value references.

    NxTrees are meant to be used as relatively shallow, hierarchical object
    containers. Primary applications are class or instance registries,
    instance caches and object indexes. A NxObject can only be found once in a
    given NxTree. While this rule isn't enforced, doing otherwise may
    alter the normal behavior of an NxTree. The NxTree implements roughly the
    same behavior as a defaultdict, where the default_factory produces further
    trees of the same type as the parent.
    Because of its recursive nature, keys of an NxTree are generally tuples,
    with each element of the tuple referring to the underlying dictionary
    at a given depth. As a result, most accessor functions expect a *key
    argument that would typically reference a tuple.
    The NxTree uses weak references from parent to child nodes. In order to
    keep the structure together, nodes -which may either be NxTrees or leaf
    nodes, themselves either NxCells or an NxObject, hold strong references to
    their parent node. This is implemented by restricting admissible value
    assignments to either NxRegistries or NxObjects.
    """

    def __init__(self):
        """To prevent errors, NxTrees cannot be passed items upon __init__."""
        super().__init__()

    # Hashability

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __ne__(self, other):
        return id(self) != id(other)

    # Regular dict method accessors

    def dictkeys(self):
        return WeakValueDictionary.keys(self)

    def dictvalues(self):
        return WeakValueDictionary.values(self)

    def dictitems(self):
        return WeakValueDictionary.items(self)

    # Administrative functions

    def child(self, key):
        """Creates a child tree of self with the same type from key.

        Note that the key passed to this function must be of single depth,
        that is, this function doesn't handle recursive insertions.

        :param key: A key to be inserted in self
        :returns: A child tree, after it's been inserted
        """
        self[key] = child_ = type(self)()
        return child_

    def copy(self):
        """Returns a deep copy of self, save for leaf values."""
        return self.__copy__()

    def __copy__(self):
        """Primitive to copy."""
        copy_ = type(self)()
        for k, v in self.dictitems():
            if isinstance(v, NxRegistry):
                copy_[k] = v.__copy__()
            else:
                copy_[k] = v
        return copy_

    # Tree accessors

    def children(self):
        """Returns a key, value pair iterable of self's direct subtrees."""
        return ((k, c) for k, c in self.dictitems() if isinstance(c, NxTree))

    def terminations(self):
        """Returns a key, value pair iterable of self's immediate objects."""
        istermination = lambda v: not isinstance(v, NxTree)
        return ((k, v) for k, v in self.dictitems() if istermination(v))

    def items(self, *depths, _parent=None):
        """Returns an iterable of key, obj pairs with self's content.

        This function returns the keys and value of leaf nodes, either
        across the entire tree, or for given depths.
        _parent is only used in the recursion to stack higher-level keys.
        :param depths: an optional sequence of integers representing tree depth
        :returns: an iterable of (key, obj) tuples
        """
        results = []
        parent = _parent or tuple()
        if not depths:
            depths = [0]
        depths = sorted(d -1 for d in depths)
        if depths[0] == -1:
            depths = depths[1:]
            for k, v in self.dictitems():
                if isinstance(v, NxTree):
                    results.extend(v.items(*depths, _parent = parent + (k, )))
                else:
                    results.append((parent + (k,), v))
        else:
            for k, v in self.dictitems():
                if isinstance(v, NxTree):
                    results.extend(v.items(*depths, _parent = parent + (k, )))
        return iter(results)

    def keys(self, *depths):
        """Returns an iterable of keys to self's leaf nodes.

        This function returns the keys from the items method.
        :param depths: an optional sequence of integers representing tree depth
        :returns: an iterable of keys
        """
        keys_, _ = zip(*self.items(*depths))
        return keys_

    def values(self, *depths):
        """Returns an iterable of the values of self's leaf nodes.

        This function returns the values from the items method.
        :param depths: an optional sequence of integers representing tree depth
        :returns: an iterable of values
        """
        _, values_ = zip(*self.items(*depths))
        return values_

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.__missing__(key)

    def __setitem__(self, key, val):
        if isinstance(val, NxRegistry):
            val._parent = self
        else:
            try:
                val._nxregistries.add(self)
            except AttributeError:
                msg = "Only NxObjects can be added as values in an NxTree."
                raise AttributeError(msg)
        super().__setitem__(key, val)

    def __delitem__(self, key):
        val = super().pop(key)  # May raise a KeyError
        if isinstance(val, NxRegistry):
            del val.parent
        else:
            val._nxregistries.remove(self)

    def __missing__(self, key):
        return self.child(key)





    def __reduce__(self):
        return NotImplemented

    def __repr__(self):
        return dict(self).__repr__()


