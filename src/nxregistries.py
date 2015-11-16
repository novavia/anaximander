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
    alter the normal behavior of an NxTree.
    The NxTree uses weak references from parent to child nodes. In order to
    keep the structure together, nodes -which may either be NxTrees or leaf
    nodes, themselves either NxCells or an NxObject, hold strong references to
    their parent node. This is implemented by restricting admissible value
    assignments to either NxRegistries or NxObjects.
    The NxTree implements the same behavior as a defaultdict, where the
    default_factory produces further trees of the same type as the parent.
    Unlike a nested dictionary, the NxTree allows random key access through
    the entire tree, using either single keys or key tuples.
    #More on this later#
    """

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __ne__(self, other):
        return id(self) != id(other)

    def __init__(self):
        """To prevent errors, NxTrees cannot be passed items upon __init__."""
        super().__init__()

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
        self[key] = value = type(self)()
        return value

    def __reduce__(self):
        return NotImplemented

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        copy_ = type(self)()
        for k, v in WeakValueDictionary.items(self):
            if isinstance(v, NxTree):
                copy_[k] = v.__copy__()
            else:
                copy_[k] = v
        return copy_

    def __repr__(self):
        return dict(self).__repr__()


