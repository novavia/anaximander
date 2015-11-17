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
    alter. However as a result an NxObject cannnot have multiple simultaneous
    registrations in the same NxRegistry.
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
    instance caches and object indexes. The NxTree implements roughly the
    same behavior as a defaultdict, where the default_factory produces further
    trees of the same type as the parent.
    The terminology for NxTrees is as follows:
    * A tree has levels corresponding to various depths.
    * Key generally refers to a single key that enables access from one level
    to the next. In order to span multiple levels, we use tuples of keys
    which together form paths. However an exception to this rule is that
    the built-in method keys actually return paths to the tree's values.
    * Nodes refer exclusively to (path, value) pairs where value is another
    NxTree -hence 'leaf nodes' are actually not considered nodes.
    * Leaves refer to (path, value) pairs where value is not an NxTree.
    * Branches refer to (key, value) pairs where value is an NxTree.
    * Twigs refer to (key, value) pairs where value is not an NxTree.
    * Basically, Nodes and Leaves are recursive properties while Branches
    and Twigs are not.
    The NxTree uses weak references to store its values. In order to
    keep the structure together, values that are inserted hold strong
    references to their parent node.
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
        return dict(WeakValueDictionary.items(self)).keys()

    def dictvalues(self):
        return dict(WeakValueDictionary.items(self)).values()

    def dictitems(self):
        return dict(WeakValueDictionary.items(self)).items()

    # Administrative functions

    def child(self, key):
        """Creates a child tree of self with the same type from key.

        :param key: A key to be inserted in self.
        :returns: A child tree, after it's been inserted.
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

    def _effective_root(self):
        """Returns the shallowest node with more than one branch / twig.

        The effective root is different from the root if the tree has
        a single branch and no twigs.
        """
        if len(self) == 1:
            k, v = list(self.dictitems)[0]
            if isinstance(v, NxTree):
                path, root = v._effective_root(self)
                return (k, ) + path, root
        return (), self

    # Tree insertion and removal

    def register(self, obj, *path):
        """Registers obj with given insertion path.

        :param obj: An NxObject or NxRegistry.
        :param *path: A path as a sequence of keys.
        """
        key, *path = path
        if len(path) == 0:
            self[key] = obj
            return
        try:
            self[key].register(obj, *path)
        except KeyError:
            self.child(key).register(obj, *path)

    def discard(self, *path):
        """Similar to a deletion, but with a recursive path.

        Note however that discard does not throw an error if the path
        does not exist.

        :param *path: An optional path as a sequence of keys.
        """
        key,*path = path
        if len(path) == 0:
            del self[key]
            return
        try:
            self[key].discard(*path)
        except KeyError:
            pass

    def unregister(self, obj, *path):
        """Unregisters obj at given insertion path.

        If the supplied path is not featured, a KeyError is raised. If it
        is found in the tree but points to an object other than obj, or to
        multiple objects, a ValueError is raised.

        :param obj: An NxObject or NxRegistry.
        :param *path: A path as a sequence of keys.
        :raises KeyError: if path is not found in self.
        :raises ValueError: if the obj, path pairing is wrong or ambiguous.
        """
        val = self.fetch(*path)
        if val == obj:
            self.discard(*path)
        else:
            raise ValueError

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

    # Tree accessors

    def branches(self):
        """Returns a key, value pair iterable of self's direct subtrees."""
        return ((k, v) for k, v in self.dictitems() if isinstance(v, NxTree))

    def twigs(self):
        """Returns a key, value pair iterable of self's direct leaves."""
        isterminal = lambda v: not isinstance(v, NxTree)
        return ((k, v) for k, v in self.dictitems() if isterminal(v))

    def _nodes(self, *depths, _path=None):
        """Primitive for nodes."""
        results = []
        path = _path or tuple()
        if not depths:
            depths = [0]
        depths = sorted(d -1 for d in depths)
        if depths[0] == -1:
            depths = depths[1:]
            for k, v in self.branches():
                results.append((path + (k, ), v))
                results.extend(v._nodes(*depths, _path = path + (k, )))
        else:
            for k, v in self.branches():
                results.extend(v._nodes(*depths, _path = path + (k, )))
        return iter(results)

    def nodes(self, *depths):
        """Returns an iterable of (path, tree) tuples found in self.

        This function returns all of the tree's nodes, optionally
        restricted to given depths. The depth of the root is set to -1 by
        convention, and the root won't be returned by nodes.
        :param *depths: optional sequence of integers representing tree depth.
        :returns: an iterable of path, tree tuples.
        """
        return self._nodes(*depths)

    def _random_key_subset(self, *path):
        """Returns a copy of nodes and leaves whose path is a superset of path.

        :param path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: a tree copy of the same type as self.
        """
        if not path:
            result = self.copy()
        else:
            result = type(self)()
            for k, v in self.dictitems():
                if k == path[0]:
                    if isinstance(v, NxTree):
                        child = v._random_key_subset(*path[1:])
                        if child: result[k] = child
                    else:
                        result[k] = v
                elif isinstance(v, NxTree):
                    child = v._random_key_subset(*path)
                    if child: result[k] = child
        return result

    def _leaves(self, _path=None):
        """Primitive for leaves."""
        results = []
        path = _path or tuple()
        for k, v in self.dictitems():
            if isinstance(v, NxTree):
                results.extend(v._leaves(_path = path + (k, )))
            else:
                results.append((path + (k, ), v))
        return iter(results)

    def leaves(self, *path):
        """Returns an iterable of path, object pairs, subset by *path.

        Leaves returns terminal objects and their accessor path. The optional
        *path specifies elements of the path, and only objects whose path
        is a subset thereof will be returned.

        :param *path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: an iterable of (path, value) items.
        """
        return self._random_key_subset(*path)._leaves()

    def items(self):
        """Returns an iterable of path, object pairs from self's leaves."""
        return dict(self._leaves()).items()

    def keys(self):
        """Returns an iterable of paths to self's leaves."""
        return dict(self._leaves()).keys()

    def values(self):
        """Returns an iterable of the values of self's leaves."""
        return dict(self._leaves()).values()

    def fetch(self, *path):
        """Returns a single item from path or raises an exception.

        Path does not need to be fully specified, i.e. an item whose path
        is a superset of the supplied path can be returned. However if no
        such item is found, a KeyError is raised. If on the other hand
        multiple items meet the path condition, a ValueError is raised.

        :param *path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :raises KeyError: if no item can be fetched.
        :raises ValueError: if multiple items are found on supplied path.
        :returns: an object stored in self.
        """
        try:
            key, values = [list(i) for i  in zip(*self.leaves(*path))]
        except ValueError: # self.leaves(*path) is empty
            raise KeyError
        if len(values) == 0:
            raise KeyError
        elif len(values) > 1:
            raise ValueError
        else:
            return values[0]

    def subset(self, *path):
        """Returns the smallest subtree that enables access to path.

        Path does not need to be fully specified, i.e. omissions are
        tolerated. The function returns an aggregage of all nodes that can
        be reached with the supplied path. If this can be done from a single
        node whose depth is greater than the root, then the subtree at that
        node is returned. As an example:
        subset(self, 'a') will return:
        self    'a' - 'a'        return    'a' - 'a'
                    - 'b'                      - 'b'
                'b' - 'a'                  'b' - 'a'
                    - 'b'
        subset(self, 'a') will return:
        self    'x' - 'a'        return    'x' - 'a'
                    - 'b'
                'y' - 'b'
        subset(self, 'a') will return:
        self    'x' - 'a'        return    'x' - 'a'
                    - 'b'                  'y' - 'a'
                'y' - 'a'

        :param *path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: an NxTree of the same type as self.
        """
        # First extract an absolute subset
        subset_ = self._random_key_subset(*path)
        # Second, truncate the access path to get to the root node
        return subset_._effective_root()[1]

    def __reduce__(self):
        return NotImplemented

    def __repr__(self):
        return dict(self.items()).__repr__()


