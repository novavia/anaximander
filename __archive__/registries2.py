#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements Anaximander registries, i.e. weak object containers.

Registries are organized as trees of Registration Nodes that hold weak
references to nxobjects. The Nodes can either be branching, that is, they
optionally store a single nxobject and reference a set of child nodes in a
weak values dictionary, or cells, in which case they hold a weak set of
nxobjects.
The registries are used for building taxonomies, keeping configuration,
information, and as instance caches or indexes.
Every Anaximander object or type has an attribute _nxregistrations which keeps
a set of strong references to the registries in which it is featured.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""



#    Registration trees are meant to be used as relatively shallow,
#    hierarchical object containers. Primary applications are class or instance registries,
#    instance caches and object indexes.


#==============================================================================
### Imports
#==============================================================================

import abc
import weakref
from itertools import chain
from weakref import WeakSet, WeakValueDictionary

from blist import weaksortedset

from ..utilities import functions as fun
from ..utilities import xprops

#==============================================================================
### Abstract base class for registration nodes.
#==============================================================================


class NxRegistrationNode(abc.ABC):
    """Abstract base class for branching and cell registration nodes.

    The base class implements the parent property, a strong reference
    to a possible parent container of the object, which is assigned to all
    but the root node of a Registry.
    Nodes can only have one parent and this has implications: the same
    node cannot be featured multiple times. If something along
    those lines is required, then the structure needs to be copied.
    nxobjects can be featured in multiple nodes, and for
    hat purpose they carry a _nxregistrations set attribute, which nodes
    can access and alter.
    """

    @xprops.cachedproperty
    def parent(self):
        return None

    @abc.abstractproperty
    def leaf(self):
        """True if self is a terminal node."""
        pass

    @abc.abstractmethod
    def register(self, obj, *path):
        """Registers obj with an optional insertion path."""
        pass

    @abc.abstractmethod
    def unregister(self, obj, *path):
        """Unregisters obj if found in self, optionally at given path."""
        pass

    @abc.abstractmethod
    def __call__(self):
        """Returns an iterable of the nxobjects stored by self."""
        pass

    def clear(self):
        """Empties the node, i.e. unregisters its content.

        Note for NxBranchingNode, this is different from the basic
        dictionary's clear method.
        """
        objects = list(self())
        for obj in objects:
            self.unregister(obj)

    def copy(self):
        """Returns a deep copy of self."""
        return self.__copy__()

    @abc.abstractmethod
    def __copy__(self):
        """The primitive to copy."""
        pass

    def _nodes(self, *depths, _path=None):
        """A primitive for nodes in NxBranchingNode.

        This is the default implementation for cell types.
        """
        yield _path, self

    def _random_key_subset(self, *path):
        """A primitive for NxBranchingNode.

        This is the default implementation for cell types.
        """
        return self

    def _effective_root(self):
        """A primitive for NxBranchingNode.

        This is the default implementation for cell types.
        """
        return (), self

    def _leaves(self, _path=None):
        """A primitive for NxBranchingNode.

        This is the default implementation for cell types.
        """
        path = _path or tuple()
        yield path, self

    def prune(self, *path):
        """Equivalent to clear for cells, reimplemented for trees."""
        self.clear()
        del self.parent


class NxBranchingNode(NxRegistrationNode, WeakValueDictionary):
    """A single-value registration node that branches out recursively.

    Though we refer to the class as a node, it can equally be considered
    a tree made up of the node itself (i.e. the root in this context), and
    the collection of nodes it can attain recursively through its branches.
    The term node was preferred to provide a common terminology and interface
    with cells, which can only be found as leaves of a registry tree structure.
    The branching node class implements a weak value dictionary to store
    access to child nodes. The structure holds together because each child
    holds a strong reference to its parent. Likewise, values stored in nodes
    are weakly referenced but the value itself holds a strong reference to the
    containing node through its _nxregistrations set.
    The terminology for trees is as follows:
    * A tree has levels corresponding to various depths. The top-level node
    is the root, and nodes without children are leaves. A node's children
    are its immediate successors in the hierarchy, while its decendants are
    all of the nodes that can be accessed from it recursively.
    * Key generally refers to a single key that enables access from one level
    to the next. In order to span multiple levels, we use tuples of keys
    which together form paths. By convention the path to a tree's root is
    the empty tuple.
    * Trees are made of nodes and the nodes can be either branching nodes,
    cell nodes or sorted cell nodes. However the latter two can only be
    leaves. Hence while branching nodes and trees are just dual designation of
    the same type of objects, cells and sorted cells are solidly nodes and
    not trees.
    * In addition to its children, a branching nodes stores a single value,
    which can be accessed by its value attribute or by a call (in which case
    and iterator is returned). This contrasts with cells that store a
    collection but don't have children.
    * Branches refer to the (key, child) tuples accessible by a node.
    * Twigs refer to branches where the child is a leaf.
    * As a result, trees / branching nodes implement the following methods:
        - value (property) is the stored value of the node;
        - keys are keys enabling access to children;
        - branches are the (key, child) tuples accessible by the node;
        - children are the child nodes of the node;
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
    * Values stored in cells are chained together with other values, such
    that childvalues, nodevalues, descendantvalues, leafvalues return iterators
    of values in which cell contents are already unpacked. The same holds
    true for items. However paths are returned without duplicates. From this
    it follows that paths and items / values generally don't have the same
    length. This is only the case if the tree contains no cell.
    * Basically, paths, nodes, descendants and leaves are recursive properties,
    while keys, children, branches and twigs are not.
    * The interface messes a bit with the dictionary interface, but the
    class implements dictkeys, dictvalues and dictitems that behave as one
    would expect.
    * The len of a node is the count of its keys, so this is unchanged
    from a dictionary.
    * Branching node also implements the properties height (maximum depth),
    nodecount (number of nodes), leafcount (number of leaves), and itemcount
    (number of stored values). A tree that only has a root node has a height
    of 1.
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
        if self.value is not None:
            self.value._nxregistrations.remove(self)
        if val is not None:
            try:
                val._nxregistrations.add(self)
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
        rval = self.value
        if rval is not None:
            yield self.value

    @property
    def leaf(self):
        """True if self has no children."""
        return len(self) == 0

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

    # Tree properties

    @property
    def height(self):
        return max(len(p) for p in self.nodekeys()) + 1

    @property
    def nodecount(self):
        return len(list(self.nodes()))

    @property
    def leafcount(self):
        return len(list(self.leaves()))

    @property
    def itemcount(self):
        return len(list(self.items()))

    # Tree accessors

    def get(self, *path):
        """Extracts node at path, or raises KeyError."""
        node = self
        while path:
            key, *path = path
            node = node[key]
        return node

    def getvalue(self, *path):
        """Returns the value stored at path, or raises KeyError."""
        return self.get(*path).value

    def getvalues(self, *path):
        """Returns a generator of values at path, or raises KeyError."""
        return self.get(*path)()

    keys = dictkeys
    children = dictvalues
    branches = dictitems

    def twigs(self):
        """Returns a key, value iterable of self's children leaves."""
        return ((k, v) for k, v in self.branches() if v.leaf)

    def childkeys(self):
        """Returns an iterable of self's keys."""
        return self.keys()

    def childitems(self):
        """Returns k, v iterable with values stored in self's children."""
        return chain(*[[(k, v) for v in n()] for k, n in self.branches()])

    def childvalues(self):
        """Returns an iterable of nxobjects stored in self's children."""
        return chain(*[n() for n in self.children])

    def _nodes(self, *depths, _path=None):
        """Primitive for nodes."""
        results = []
        path = _path or tuple()
        if not depths:
            depths = [0]
        if depths[0] == 0:
            results.append((path + (), self))
            depths = depths[1:]
        depths = sorted(d - 1 for d in depths)
        for k, n in self.branches():
            results.extend(n._nodes(*depths, _path=path + (k, )))
        return iter(results)

    def nodes(self, *depths):
        """Returns an iterable of (path, node) tuples found in self.

        This function returns all of the tree's nodes, optionally
        restricted to given depths. The depth of the root is set to 0 by
        convention.

        :param *depths: optional sequence of integers representing tree depth.
        :returns: an iterable of path, node tuples.
        """
        return self._nodes(*depths)

    def nodekeys(self, *depths):
        """Returns an iterable of paths to self's nodes."""
        return dict(self._nodes(*depths)).keys()

    def nodeitems(self, *depths):
        """Returns k, v iterable with values stored in self's nodes."""
        return chain(*[[(k, v) for v in n()] for k, n in self._nodes(*depths)])

    def nodevalues(self, *depths):
        """Returns an iterable of nxobjects stored in self's nodes."""
        return chain(*[n() for k, n in self._nodes(*depths)])

    def items(self):
        """Equivalent to nodeitems at all depths."""
        return self.nodeitems()

    def values(self):
        """Equivalent to nodevalues at all depths."""
        return self.nodevalues()

    def descendants(self):
        """Similar to nodes but ommits the root."""
        return self._nodes(*range(1, self.height))

    def descendantkeys(self):
        """An iterable of paths to self's descendants."""
        return self.nodekeys(*range(1, self.height))

    def descendantitems(self):
        """Returns k, v iterable with values stored in self's descendants."""
        return self.nodeitems(*range(1, self.height))

    def descendantvalues(self):
        """Returns an iterable of nxobjects stored in self's descendants."""
        return self.nodevalues(*range(1, self.height))

    def _leaves(self, _path=None):
        """Primitive for leaves."""
        results = []
        path = _path or tuple()
        for k, n in self.branches():
            if n.leaf:
                results.append((path + (k, ), n))
            else:
                results.extend(n._leaves(_path=path + (k, )))
        return iter(results)

    def leaves(self):
        """Returns an iterable of path, leaf node pairs.
        """
        return self._leaves()

    def leafkeys(self):
        """Returns an iterable of keys to self's leaves."""
        return dict(self._leaves()).keys()

    def leafitems(self):
        """Returns a k, v iterable of values in leaves."""
        return chain(*[[(k, v) for v in n()] for k, n in self._leaves()])

    def leafvalues(self):
        """Returns an iterable of values stored in leaves."""
        return chain(*[n() for k, n in self._leaves()])

    def _random_key_subset(self, *path):
        """Returns a copy of nodes whose path is a superset of path.

        :param path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: a tree copy.
        """
        if not path:
            result = self.copy()
        else:
            result = type(self)()
            for k, n in self.branches():
                if k == path[0]:
                    result[k] = n._random_key_subset(*path[1:])
                else:
                    result[k] = n._random_key_subset(*path)
        return result

    def _effective_root(self, *path):
        """Returns node that meet effective root conditions.

        This returns either the shallowest node that has multiple branches,
        or the shallowest node on a single branch that is accessible through
        a superset of path.
        For instance, if 'a' gets specified and a branch with key 'a' is found
        with no branching from the root, then the corresponding node is the
        effective root.
        if 'a', 'b' is passed, it would have to be a node that is accessed
        with 'a' somewhere on its path, on a 'b' branch.
        """
        if len(self) == 1 and path:
            k, v = list(self.branches())[0]
            if k == path[0]:
                path = path[1:]
            return v._effective_root(*path)
        return self

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

        An important warning is that even though nodes that are antecedent
        to the path may be copied into subset, their stored values are not!
        Only values stored downstream from the specified path are stored
        in the subset.

        :param *path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: a tree of the same type as self.
        """
        # First extract an absolute subset
        subset_ = self._random_key_subset(*path)
        # Second, truncate the access path to get to the root node
        return subset_._effective_root(*path)

    def fetchkeys(self, *path):
        """Returns an iterable of paths in self, subset by *path.

        Path does not need to be fully specified, i.e. a path whose path
        is a superset of the supplied path can be returned.

        :param *path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: an iterable of paths in self.
        """
        return self.subset(*path).nodekeys()

    def fetchitems(self, *path):
        """Returns a k, v iterable of values that meet the path specification.

        Path does not need to be fully specified, i.e. a value whose path
        is a superset of the supplied path can be returned.

        :param *path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: an iterable of (path, nxobject) in self.
        """
        return self.subset(*path).nodeitems()

    def fetchvalues(self, *path):
        """Returns an iterable of registrees that meet the path specification.

        Path does not need to be fully specified, i.e. an object whose path
        is a superset of the supplied path can be returned.

        :param *path: an optional path specification. Omissions of parts of
            the path are tolerated.
        :returns: an iterable of nxobjects stored in self.
        """
        return self.subset(*path).nodevalues()

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
            key, values = [list(x) for x in zip(*self.fetchvalues(*path))]
        except ValueError:  # self.fetchvalues(*path) is empty
            raise KeyError
        if len(values) == 0:
            raise KeyError
        elif len(values) > 1:
            raise ValueError
        else:
            return values[0]

    # Tree insertion and removal

    def __setitem__(self, key, node):
        try:
            self[key].prune()
        except KeyError:
            pass
        if isinstance(node, NxRegistrationNode):
            node._parent = self
        else:
            msg = "Only NxRegistrationNode objects can be inserted into a " + \
                  "registry tree structure."
            raise TypeError(msg)
        super().__setitem__(key, node)

    def insert(self, node, key, *path):
        """Insert a new node at provided path as (key,) + path.

        Note that insert will overwrite existing nodes.

        :param node: An NxRegistration Node.
        :param key: A required key.
        :param *path: The remainder of the insertion path.
        """
        self.prune(*((key,) + path))
        if len(path) == 0:
            self[key] = node
            return
        self.child(key).insert(node, *path)

    def child(self, key, *values, type_=None):
        """Creates a child node of self of type type_ and stores values.

        If kwarg-only type_ is not supplied, then type(self) is used.
        If there are more than one value and the child's type is
        NxBranchingNode, a TypeError is raised.

        :param key: A key to be inserted in self.
        :param values: An optional list of values stored in the child node.
        :param type_: An optional node type for the child.
        :raises TypeError: If multiple values are passed to a branching node.
        :returns: A child node, after it's been inserted.
        """
        type_ = fun.get(type_, type(self))
        self[key] = child_ = type_(*values)
        return child_

    def prune(self, *path):
        """Removes the node at path and its descendants.

        :param *path: An optional path as a sequence of keys.
        :raises KeyError: If path is not valid.
        """
        root = self.get(*path)
        _, nodes = [list(x) for x in zip(*root.nodes())]
        for n in nodes:
            n.clear()
            del n.parent

    def discard(self, *path):
        """Similar to prune but silences KeyErrors."""
        try:
            self.prune(*path)
        except KeyError:
            pass

    def __delitem__(self, key):
        """Equivalent to pruning for a child."""
        self[key].prune()  # May raise a KeyError

    def sever(self, depth=0):
        """Prunes all nodes whose depth is greater than the supplied depth.

        :param depth: An integer 0 or greater.
        """
        for n in self.nodekeys(depth + 1):
            n.prune()

    def pop(self, key, *args):
        return NotImplemented

    def popitem(self):
        return NotImplemented

    def setdefault(self, key, default=None):
        return NotImplemented

    def update(self, dict=None, **kwargs):
        return NotImplemented

# Registration / unregistration functions

    def register(self, obj, *path, type_=None):
        """Registers obj with given insertion path.

        :param obj: An nxobject instance.
        :param *path: A path as a sequence of keys.
        :param type_: The type of node to be created, if needed.
        """
        if len(path) == 0:
            self.value = obj
            return
        key, *path = path
        try:
            self[key].register(obj, *path)
        except KeyError:
            if path:
                self.child(key).register(obj, *path, type_=type_)
            else:
                self.child(key, obj, type_=type_)

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

    # Administrative functions

    def __copy__(self):
        """Primitive to copy."""
        copy_ = type(self)(self.value)
        for k, n in self.branches():
                copy_[k] = n.__copy__()
        return copy_

    def __reduce__(self):
        return NotImplemented

    def __repr__(self):
        cls = type(self).__name__
        if not self:
            return '{cls}()'.format(**locals())
        leaves = fun.spformat(self.leafcount, 'leaf', 'leaves')
        height = 'height={0}'.format(self.height)
        items = fun.spformat(self.itemcount)
        return '{cls}({leaves}, {height}, {items})'.format(**locals())
