#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements Anaximander registries, i.e. weak object containers.

The main two registry types are NxCell (a weak set) and NxTree (a recursive
weak values dictionary). The registries are used for building taxonomies,
keeping configuration information, and as instance caches or indexes.
Every Anaximander object or type has an attribute _nxregistries which keeps
a set of strong references to the registries in which it is featured.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import abc
from itertools import chain
from weakref import WeakSet, WeakValueDictionary

from blist import weaksortedset

from ..utilities import functions as fun
from ..utilities import xprops

#==============================================================================
### Abstract base class
#==============================================================================


class NxRegistry(abc.ABC):
    """Abstract base class for NxCell and NxTree.

    The base class implements the parent property, a strong reference
    to a possible parent container of the object, as is the case with
    a subtree or a cell that is a leaf node of a tree.
    NxRegistries can only have one parent and this has implications: the same
    NxRegistry cannot be a node to multiple NxTrees. If something along
    those lines is required, then the structure needs to be copied.
    nxobjects can be featured in multiple NxRegistries, and for
    hat purpose they carry a _nxregistries set attribute, which NxRegistry
    can access and alter. However as a result an nxobject instance cannnot
    reliably have multiple simultaneous registrations in the same NxRegistry.
    """

    @xprops.cachedproperty
    def parent(self):
        return None

    @abc.abstractmethod
    def register(self, obj, *path):
        """Registers obj with an optional key path."""
        pass

    @abc.abstractmethod
    def unregister(self, obj, *path):
        """Unregisters obj at given insertion path."""
        pass

#==============================================================================
### Concrete classes
#==============================================================================


class NxCell(WeakSet, NxRegistry):
    """A weak container of nxobject instances."""

    def __init__(self):
        super().__init__()

    # Hashability

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __ne__(self, other):
        return id(self) != id(other)

    # Access methods

    def add(self, item):
        """Overwrites the default add method."""
        try:
            item._nxregistries.add(self)
        except AttributeError:
            msg = "Only nxobject instances can be added to an NxCell."
            raise TypeError(msg)
        super().add(item)

    def remove(self, item):
        super().remove(item)
        item._nxregistries.remove(self)

    def discard(self, item):
        try:
            self.remove(item)
        except KeyError:
            pass

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

    def register(self, obj):
        """Adds an object to the cell.

        :param obj: An nxobject instance.
        """
        self.add(obj)

    def unregister(self, obj):
        """Removes object from the cell.

        :param obj: An nxobject instance.
        """
        self.discard(obj)

    def copy(self):
        """Returns a deep copy of self save for non-registry objects."""
        return self.__copy__()

    def __copy__(self):
        """Primitive to copy."""
        copy_ = type(self)()
        for item in self:
                copy_.add(item)
        return copy_

    def __repr__(self):
        cls = type(self).__name__
        if not self:
            return '{cls}()'.format(**locals())
        content = fun.spformat(self)
        return '{cls}({content})'.format(**locals())


# weaksortedset does not inherit from WeakSet, and therefore NxSortedCell
# does not inherit from NxCell. As a result, all the methods are fully
# rewritten. This could be done more sparingly but given the size of the
# code that seemed like the safest and most efficient route.

class NxSortedCellType(abc.ABCMeta):
    """The metaclass for NxSortedCell."""
    __cache__ = WeakValueDictionary()  # type cache

    def __new__(mcl, name, bases, namespace, **kwargs):
        """Customized to intercept key and provide caching / retrieving."""
        try:
            key = namespace['__key__']
        except KeyError:
            msg = "NxSortedCell type requires a __key__ atrribute."
            raise ValueError(msg)
        try:
            return mcl.__cache__[key]
        except KeyError:
            cell_type = super().__new__(mcl, name, bases, namespace)
            mcl.__cache__[key] = cell_type
            return cell_type

    @classmethod
    def _from_key(mcl, key):
        """Makes a NxSortedCell type from a sort key."""
        base_type = mcl.__cache__[None]
        name, bases, namespace = fun.metargs(base_type)
        bases = (base_type,)
        namespace['__key__'] = key
        return mcl(name, bases, namespace)

    def __getitem__(cls, key):
        """Enables retrieving types by sort key."""
        try:
            return cls.__cache__[key]
        except KeyError:
            return cls._from_key(key)


class NxSortedCell(weaksortedset, NxRegistry, metaclass=NxSortedCellType):
    """A weak sorted container of nxobject instances."""
    __key__ = None  # A default sort key

    def __init__(self, key=None):
        """Specify a sort key, otherwise takes class' default."""
        super().__init__(key=fun.get(key, self.__key__))

    # Hashability

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __ne__(self, other):
        return id(self) != id(other)

    # Access methods

    def __getitem__(self, index):
        """A slice returns a weaksortedset rather than a NxSortedCell."""
        if isinstance(index, slice):
            rv = weaksortedset()
            rv._blist = self._blist[index]
            rv._key = self._key
            return rv
        return super().__getitem__(index)

    def add(self, item):
        """Overwrites the default add method."""
        try:
            item._nxregistries.add(self)
        except AttributeError:
            msg = "Only nxobject instances can be added to an NxCell."
            raise TypeError(msg)
        super().add(item)

    def remove(self, item):
        if not item in self:
            raise KeyError
        self.discard(item)

    def discard(self, item):
        if not item in self:
            return
        super().discard(item)
        item._nxregistries.remove(self)

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

    def register(self, obj):
        """Adds an object to the cell.

        :param obj: An nxobject instance.
        """
        self.add(obj)

    def unregister(self, obj):
        """Removes object from the cell.

        :param obj: An nxobject instance.
        """
        self.discard(obj)

    def copy(self):
        """Returns a copy of self."""
        return self.__copy__()

    def __copy__(self):
        """Primitive to copy."""
        copy_ = type(self)()
        for item in self:
                copy_.add(item)
        return copy_

    def __repr__(self):
        cls = type(self).__name__
        if not self:
            return '{cls}()'.format(**locals())
        content = fun.spformat(self)
        return '{cls}({content})'.format(**locals())


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
    which together form paths. By convention the path to a tree's root is
    the empty tuple.
    * Trees are made of nodes, but node is only a contextual designation for
    the descendants of a given tree since there is no node class. Nodes can
    be any type of NxRegistry, i.e. NxCell, NxSortedCell or NxTree.
    * In addition to its children, an NxTree stores a single value, which can
    be accessed by its value attribute or by a call.
    * Leaves refer to nodes without children.
    * Branches refer to the (key, child) tuples accessible by a root node.
    * Twigs refer to branches where the child is a leaf.
    * As a result, NxTree implements the following methods:
        - value (property) is the stored value of the root node;
        - keys are the known keys of the root node;
        - branches are the (key, child) tuples accessible by the root node;
        - children are the children of the root node;
        - childpaths, childitems and childvalues mimic keys, items and values
        of a dictionary, where values are the values stored by children nodes.
        childpaths is equivalent to keys;
        - nodes are all of the (path, node) tuples of the tree, including the
        root;
        - nodepath, nodeitems, nodevalues mimic keys, items and values of a
        dictionary, where values are the values stored by nodes;
        - descendants are (path, node) tuples of the tree, minus the root;
        - descendantpaths, descendantitems, descendantvalues mimic keys, items
        and values of a dictionary, where values are values stored by
        descendants;
        - nodes and descendants, as well as their dict-like associated methods
        accept an optional *depths argument to restrict the results to
        specified depths;
        - items and values are aliases for nodeitems and nodevalues. Paths is
        an alias for nodepath.
        - leaves are (path, node) tuples of the tree where node is childless.
        Note that this may include the root;
        _ leafpaths, leafitems, leafvalues mimic keys, items and values of a
        dictionary, where values are the values stored by leaf nodes;
    * All the items / values methods exclude None values from their results.
    * Cells are returned as nodes by branches, children, nodes, descendants
    and leaves. Hence when iterating or traversing, one must check for the
    type of the nodes.
    * Values stored in cells are chained together with NxTree values, such
    that childvalues, nodevalues, descendantvalues, leafvalues return iterators
    of values in which cell contents are already unpacked. The same holds
    true for items. However paths are returned without duplicates. From this
    it follows that paths and items / values generally don't have the same
    length. This is only the case if the Tree contains no cell.
    * Basically, paths, nodes, descendants and leaves are recursive properties,
    while keys, children, branches and twigs are not.
    * The interface messes a bit with the dictionary interface, but
    NxTree implements dictkeys, dictvalues and dictitems that behave as one
    would expect.
    * The len of an NxTree is the count of its keys, so this is unchanged
    from a dictionary.
    * NxTree also implments the properties height (maximum depth), nodecount
    (number of nodes), leafcount (number of leaves), and itemcount (number of
    stored values).
    The NxTree uses weak references to store its nodes and values. In order to
    keep the structure together, stored values hold strong
    references to their node, and children nodes hold strong references to
    their parent.
    """

    def __init__(self, value=None):
        """To prevent errors, NxTrees cannot be passed items upon __init__."""
        super().__init__()
        if value is not None:
            self.value = value

    @xprops.weakproperty
    def value(self):
        return None

    @value.setter
    def value(self, val=None):
        if self._value is not None:
            self._value._nxregistries.remove(self)
        if val is not None:
            try:
                val._nxregistries.add(self)
            except AttributeError:
                msg = "Only nxobjects can be stored in an NxRegistry."
                raise TypeError(msg)
            self._value = weakref.ref(val)
        else:
            self._value = None

    @value.deleter
    def value(self):
        self.value = None

    def __call__(self):
        return self._value

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

    def child(self, key, value=None):
        """Creates a child tree of self with the same type from key.

        :param key: A keremovey to be inserted in self.
        :param value: An optional value stored in the child's root node.
        :returns: A child tree, after it's been inserted.
        """
        self[key] = child_ = type(self)(value)
        return child_

    def copy(self):
        """Returns a deep copy of self."""
        return self.__copy__()

    def __copy__(self):
        """Primitive to copy."""
        copy_ = type(self)(self.value)
        for k, v in self.dictitems():
                copy_[k] = v.__copy__()
        return copy_

#    def _effective_root(self):
#        """Returns the shallowest path, node with more than one branch.
#
#        The effective root is different from the root if the tree has
#        a single branch and no twigs.
#        """
#        if len(self) == 1:
#            k, v = list(self.dictitems())[0]
#            if isinstance(v, NxTree):
#                path, root = v._effective_root()
#                return (k, ) + path, root
#        return (), self

# TODO: make sure this works.
    def clear(self):
        for path, obj in list(self.items()):
            self.unregister(obj, path)

    # Tree insertion and removal

    def register(self, obj, *path):
        """Registers obj with given insertion path.

        :param obj: An nxobject instance.
        :param *path: A path as a sequence of keys.
        """
        if len(path) == 0:
            self.value = obj
            return
        key, *path = path
        try:
            self[key].register(obj, *path)
        except KeyError:
            self.child(key).register(obj, *path)

# TODO: this implementation only works with complete paths, may want partial paths
    def unregister(self, obj, *path):
        """Unregisters obj at given insertion path.

        If the supplied path is not featured, a KeyError is raised. If it
        is found in the tree but obj is not registered there, the function
        dies silently and no exception gets raised.

        :param obj: An nxobject instance.
        :param *path: A path as a sequence of keys.
        :raises KeyError: if path is not found in self.
        """
        if len(path) == 0:
            if self.value == obj:
                del self.value
            return
        key, *path = path
        self[key].unregister(obj, *path)

    def __setitem__(self, key, node):
        if isinstance(node, NxRegistry):
            node._parent = self
        else:
            msg = "Only NxRegistries can be inserted into an NxTree."
            raise TypeError(msg)
        super().__setitem__(key, node)

    def __delitem__(self, key):
        node = super().pop(key)  # May raise a KeyError
        del node.value
        del node.parent

    def prune(self, *path):
        """Removes a node and descendants at provided path.

        :param *path: An optional path as a sequence of keys.
        :raises KeyError: if path is not found in self.
        """
        if len(path) == 0:
            self.clear()
            return
        key, *path = path
        self[key].remove(*path)

    def discard(self, *path):
        """Similar to a deletion, but with a recursive path.

        Note however that discard does not throw an error if the path
        does not exist.

        :param *path: An optional path as a sequence of keys.
        """
        if len(path) == 0:
            key, *path = path
        try:
            if len(path) == 0:
                del self[key]
                return
            else:
                self[key].discard(*path)
        except KeyError:
            pass

    def pop(self, key, *args):
        return NotImplemented

    def popitem(self):
        return NotImplemented

    def setdefault(self, key, default=None):
        return NotImplemented

    def update(self, dict=None, **kwargs):
        return NotImplemented

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
        depths = sorted(d - 1 for d in depths)
        if depths[0] == -1:
            depths = depths[1:]
            for k, v in self.branches():
                results.append((path + (k, ), v))
                results.extend(v._nodes(*depths, _path=path + (k, )))
        else:
            for k, v in self.branches():
                results.extend(v._nodes(*depths, _path=path + (k, )))
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
                        if child:
                            result[k] = child
                    else:
                        result[k] = v
                elif isinstance(v, NxTree):
                    child = v._random_key_subset(*path)
                    if child:
                        result[k] = child
        return result

    def _leaves(self, _path=None):
        """Primitive for leaves."""
        results = []
        path = _path or tuple()
        for k, v in self.dictitems():
            if isinstance(v, NxTree):
                results.extend(v._leaves(_path=path + (k, )))
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
            key, values = [list(i) for i in zip(*self.leaves(*path))]
        except ValueError:  # self.leaves(*path) is empty
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
        cls = type(self).__name__
        if not self:
            return '{cls}()'.format(**locals())
        paths, values = zip(*self._leaves())
        max_depth = max(len(p) for p in paths) - 1
        content = fun.spformat(values, 'leaf', 'leaves')
        return '{cls}({content}, max_depth={max_depth})'.format(**locals())


class NxCellTreeType(abc.ABCMeta):
    """The metaclass for NxCellTree."""
    __cache__ = WeakValueDictionary()  # type cache

    def __new__(mcl, name, bases, namespace, **kwargs):
        """Customized to intercept key and provide caching / retrieving."""
        try:
            cell_type = namespace['__cell__']
        except KeyError:
            msg = "NxCellTree type requires a __cell__ atrribute."
            raise ValueError(msg)
        try:
            return mcl.__cache__[cell_type]
        except KeyError:
            tree_type = super().__new__(mcl, name, bases, namespace)
            mcl.__cache__[cell_type] = tree_type
            return tree_type

    @classmethod
    def _from_cell_type(mcl, cell_type):
        """Makes a NxCellTree type from a cell type."""
        base_type = mcl.__cache__[NxCell]
        name, bases, namespace = fun.metargs(base_type)
        bases = (base_type,)
        namespace['__cell__'] = cell_type
        return mcl(name, bases, namespace)

    def __getitem__(cls, cell_type):
        """Enables retrieving types by cell_type."""
        try:
            return cls.__cache__[cell_type]
        except KeyError:
            return cls._from_cell_type(cell_type)


class NxCellTree(NxTree, metaclass=NxCellTreeType):
    """An NxTree whose leaves are NxCells."""
    __cell__ = NxCell  # NxCell type

    def leaf(self, key):
        """Creates an NxCell as a direct leaf of self.

        :param key: A key to be inserted in self.
        :returns: An NxCell, after it's been inserted.
        """
        self[key] = cell = self.__cell__()
        return cell

    # Tree insertion and removal

    def __setitem__(self, key, val):
        if not isinstance(val, NxRegistry):
            msg = "NxRegistries are the only admissible values in NxCellTree."
            raise AttributeError(msg)
        super().__setitem__(key, val)

    def register(self, obj, *path):
        """Registers obj with given insertion path.

        :param obj: An nxobject instance.
        :param *path: A path as a sequence of keys.
        """
        key, *path = path
        if len(path) == 0:
            try:
                self[key].register(obj)
            except KeyError:
                self.leaf(key).register(obj)
        else:
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
        key, *path = path
        try:
            if len(path) == 0:
                del self[key]
                return
            else:
                self[key].discard(*path)
        except KeyError:
            pass

    def unregister(self, obj, *path):
        """Unregisters obj at given insertion path.

        If the supplied path is not featured, a KeyError is raised. If it
        is found in the tree but points to a cell that doesn't contain obj,
        a KeyError is raised as well.

        :param obj: An nxobject instance.
        :param *path: A path as a sequence of keys.
        :raises KeyError: if path not found in self or obj not found at path.
        :raises ValueError: if path is ambiguous.
        """
        cell = self.fetch(*path)
        cell.unregister(obj)

    # Note that the interface below is a stark departure from a dictionary
    # interface: items and values return nxobject instances stored in the leaf
    # cells, while keys only return the set of paths leading to those
    # cells. In particular, keys and items / values will usually have
    # different lengths.

    def items(self):
        """Returns an iterable of path, object pairs from self's leaves."""
        def expand(path, cell):
            """Returns an iterable of (path, item) tuples."""
            return ((path, item) for item in cell)
        return chain(*[expand(*leaf) for leaf in self._leaves()])

    def keys(self):
        """Returns an iterable of paths to self's leaves."""
        return dict(self._leaves()).keys()

    def values(self):
        """Returns an iterable of the values stored in self's leaves."""
        vals = chain(*[leaf[1] for leaf in self._leaves()])
        return dict(enumerate(vals)).values()

    def fetch(self, *path):
        """Returns a single cell from path or raises an exception.

        Path does not need to be fully specified, i.e. a cell whose path
        is a superset of the supplied path can be returned. However if no
        such cell is found, a KeyError is raised. If on the other hand
        multiple cells meet the path condition, a ValueError is raised.

        :param *path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :raises KeyError: if no item can be fetched.
        :raises ValueError: if multiple items are found on supplied path.
        :returns: a leaf cell stored in self.
        """
        try:
            key, values = [list(i) for i in zip(*self.leaves(*path))]
        except ValueError:  # self.leaves(*path) is empty
            raise KeyError
        if len(values) == 0:
            raise KeyError
        elif len(values) > 1:
            raise ValueError
        else:
            return values[0]

    def __repr__(self):
        cls = type(self).__name__
        if not self:
            return '{cls}()'.format(**locals())
        paths, cells = zip(*self._leaves())
        max_depth = max(len(p) for p in paths) - 1
        cells_ = fun.spformat(cells, 'cell')
        items_ = fun.spformat(list(chain(*cells)))
        content = ', '.join((cells_, items_))
        return '{cls}({content}, max_depth={max_depth})'.format(**locals())
