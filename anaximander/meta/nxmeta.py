#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines Anaximander nxmeta, the type interpreter.

A foundational and distinctive feature of Anaximander is its dynamic
type creation capabilities. Traditional object-oriented programming patterns
define classes as containers of attributes and methods that serve as
templates for individual objects. Anaximander is a heavily metaprogramming-
oriented framework, and it treats types themselves very much as objects.
The reason for this is that it is typical in natural sciences to deal
will collection of objects that share many characteristics but not all.
It is possible to create base classes and let instances specifiy attributes
along which they differ from one another, but if there are thousands or
more instances of a given class, this can start having an impact on
memory footprint. Further, if the fact to exhibit certain attributes need
to lead to implementing different methods, then there is no effective
replacement for creating as many types as is required. That can be
a programming burden.
Anaximander offers three basic mechanisms to improve efficiency and
productivity when dealing with collections of items that are closely
related but have slightly different attributes and methods:
* First, there is the Assembler, which is simply a wrapper for programmatic
type creation. Assembler invokes the new_class function of the standard
library types module, but it adds to it a behavior similar to partial
in the functools module. Hence types can be specified by iterative calls.
The Assembler doesn't per se solve any of the aforementionned problems,
but it provides a framework for dynamic class creation. An Assembler
object is a class factory.
* Second is the notion of archetype and prototype. An archetype is a class
that has 'unfilled' class attributes. For instance, a Car could be an
archetype, which has a 'brand' attribute set to None. From that archetype,
we could create types for Tesla or Mercedes cars, each of which has
a brand set to 'Tesla' and 'Mercedes', respectively. This is more efficient
than passing a brand name to every instance, especially when dealing
with thousands or more instances. However we may clearly not want to
explicitly declare a Tesla class and Mercedes class -that would be OK
with two brands, but there could be dozens. Anaximander solves this
problem by decorating Car as an archetype with a 'meta attribute' that is
the 'brand'. An archetype lets the programmer define so-called 'foretypes'
by providing concrete values for the meta attributes. A particular set
of meta attributes is stored as a genotype, of which the foretype inherits
in addition to the archetype. The collection of foretypes that share the same
archetype is called a Clade, and it is cached in a particular type of registry
held by the archetype, called a Cladogram. In the example provided there is
only one meta attribute, but there could be several, and the Cladogram can
then adopt a nested structure in which each level of depth corresponds to a
particular meta attribute. Finally, a prototype is simply an archetype whose
meta attributes are mandatorily required in order to create foretypes. By
contrast, a general archetype may create foretypes for which only part of
the meta attributes are concretely defined.
* Third is the notion of basetype and overtype. A basetype is a type that the
programmer intends to define a common set of attributes and behaviors for
a family of objects, knowing that some of these objects will implement
additional attributes and behaviors. For instance, our class Car from above
could be designed as a basetype and feature a body, an engine and four wheels.
However we know that certain options such as an ABS system or a sunroof
may be found on certain cars. Again, rather that adding those attributes
and possible derived methods to each individual instances, it is more
efficient to create types for the different combination of available
options -this may cease to be true when the combinations become large
compared to the number of instances, so that's always a judgment call on the
part of the programmer. Traditionally, such a design would be implemented
with the use of mixin classes. Anaximander simply offers a programmatic
mechanism to obtain an equivalent outcome without requiring declarative
instructions. An overtype is a type that adds a 'phenotype' of attributes
to a basetype. In practice a phenotype is a mixin class from which the
overtype inherits, but that is simply instantiated by passing data
descriptors in a given sequence.
Because of another Anaximander metaprogramming feature, which is to
dynamically supplement a class with methods based on its descriptors,
the overtype may also implement methods not found in the basetype.
Similarly to an archetype, a basetype caches the overtypes that it engenders.
The basetype and its overtypes define a Species, and the basetype registers
Species members in a Phenogram, which is essentially a dictionary whose
keys are a hash of the members' phenotypes.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

from collections import ChainMap, OrderedDict
from functools import partial
from inspect import getmodule
from itertools import count
import sys
import types

from .. import nxtype, nxobject
from ..registries import registries as reg, folios as fol
from ..utilities.xprops import settablecachedproperty
from ..utilities.functions import lmap, get
from ..utilities.cmpmixin import ComparableMixin

#==============================================================================
### Exception class
#==============================================================================


class MetaError(Exception):
    """Metaprogrammation exception class."""
    pass

#==============================================================================
### Assembler machinery.
#==============================================================================


class Assembler(nxobject):
    """Assemblers are callables that return types.

    The Assembler class as a whole is a type factory. Any Assembler
    instance can be made to return any kind of classes, similarly to
    the new_class function defined in the types standard library.
    Assembler combines this functionality with a behavior akin to
    partial in the functools standard library, such that Assembler
    instances can be built iteratively from multiple calls.
    Assembler can automtically name a class if the metaclass used to
    instantiate it features a classmethod __baptize__, which takes
    bases, patch and **keys as arguments and returns a name.
    """
    # Class defaults, which may be overriden by subclasses.
    bases = ()  # type bases.
    metaclass = None  # metaclass used to create types.
    patch = None  # optional namespace patch.
    slots = None  # Optional tuple of __slot__ strings.
    keys = None  # optional type keys, passed to the metaclass.

    def __init__(self, *bases, metaclass=None, patch=None, slots=None, **keys):
        """Initializes an Assembler. All arguments are optional.

        :param *bases: a sequence of base classes.
        :param metaclass: an optional metaclass.
        :param patch: a dictionary of additional class attributes & methods.
        :param slots: a tuple of __slots__ strings.
        :param **keys: type keys / metaclass keyword arguments, as applicable.
        :returns: an Assembler instance.
        """
        self.keys = {}
        kwargs = ChainMap(keys, lmap('metaclass', 'patch', 'slots'))
        self.update(*bases, **kwargs)

    def update(self, *bases, metaclass=None, patch=None, slots=None, **keys):
        """Updates an Assembler instance with new / additional parameters.

        Note that all of the parameters except keys update through overwriting.
        One cannot, say, add a base class or augment an existing patch with an
        additional method. For this, a new bases tuple or patch dictionary
        need to be created then passed to the update method.
        Keys work the other way round, with a dictionary update.
        """
        if bases:
            self.bases = bases
        if metaclass is not None:
            self.metaclass = metaclass
        if patch is not None:
            self.patch = patch
        if slots is not None:
            self.slots = slots
        self.keys.update(keys)

    def __call__(self, name=None, **kwargs):
        """The main interface, returns either a type or an updated Assembler.

        Admissible kwargs include bases, metaclass, patch and slots, which have
        the result of updating those instance variables locally to the call.
        Additional kwargs are treated as type keys updates.
        If the attempt to create a class from the supplied argument
        fails with a TypeError, an Assembler instance  with updated arguments
        is returned in lieu of a type.
        """
        bases = kwargs.pop('bases', self.bases)
        metaclass = kwargs.pop('metaclass', self.metaclass)
        patch = kwargs.pop('patch', self.patch)
        slots = kwargs.pop('slots', self.slots)
        keys = self.keys.copy()
        if slots is not None:
            keys['slots'] = slots
        keys.update(kwargs)
        metarg = {'metaclass': metaclass} if metaclass else {}
        kwds = ChainMap(metarg, keys)
        # Callable that returns function's return value if not a type.
        assembler = lambda: type(self)(*bases, patch=patch, **kwds)
        if not name:
            # Find the appropriate metaclass by calling prepare with mock name
            metaclass, _, _ = types.prepare_class('mock', bases, kwds)
            if not hasattr(metaclass, '__baptize__'):
                return assembler()
            name = metaclass.__baptize__(bases, **keys)
        if patch is not None:
            exec_body = lambda ns: ns.update(patch)
        else:
            exec_body = lambda ns: ns
        try:
            cls = types.new_class(name, bases, kwds, exec_body)
        # Failure to create new class, normally attributable to insufficient
        # attribute specifications.
        except MetaError:
            return assembler()
        cls.__module__ = getmodule(sys._getframe(1))
        return cls

#==============================================================================
### Base descriptors
#==============================================================================


class DescriptorRegistry(reg.NxRegistry):
    """A registry of nxdescriptors organized by type."""
    __root__ = fol.NxPortFolio
    __layers__ = [('type', fol.NxScroll)]


class nxdescriptor(nxobject, ComparableMixin):
    """The base Anaximander descriptor class."""
    __count__ = count()

    def __init__(self):
        """Descriptors receive an id to retrieve instantiation order."""
        self.descriptor_id = next(self.__count__)

    @settablecachedproperty
    def name(self):
        """Optional name set by nxmeta."""
        return None

    def _cmpkey(self):
        return self.descriptor_id

    def __metainit__(self, cls):
        """Called by nxmeta in its __init__ method."""
        pass


class typeattribute(nxdescriptor):
    """Identifies type attributes in archetypes.

    Note that this is technically not a true descriptor. It is portrayed
    as such because it serves the purpose of identifying specific
    attributes in archetype declarations. Once the typeattribute has
    been registered, it is replaced by a regular class variable.
    """

    def __init__(self, default=None, *, dtype=None):
        """Specifies optional default value and data type."""
        super().__init__()
        self.default = default
        self.dtype = dtype
        self.cache = None  # Name of storage attribute in host class

    def __get__(self, obj, objtype=None):
        return getattr(objtype, self.cache)

    def __set__(self, obj, value):
        raise AttributeError("Can't set type attribute.")

    def __delete__(self, obj):
        raise AttributeError("Can't delete type attribute.")

    def __metainit__(self, cls):
        """Called by the archetype that defines the attribute."""
        setattr(cls, self.cache, self.default)

    def foreset(self, cls, value=None):
        """Sets the attribute in a foretype."""
        value = get(value, self.default)
        if self.dtype is not None:
            if not isinstance(value, self.dtype):
                raise TypeError
        setattr(cls, self.cache, value)


class typemethod(classmethod, nxdescriptor):
    """A classmethod that provides implementation to a metaclass method.

    For instance, nxmeta defines __getitem__ to allow bracket notations
    on types. The actual implementation of __getitem__ is kept at the type
    level in the method __typegetitem__, which is a typemethod. Compared to
    a classmethod, a type method simply checks that there is a registered
    metaclass method that is targeted by it -e.g. __getitem__ in the proposed
    example. Of course such check is optional, and a classmethod decoration
    would work in most cases but the typemethod makes the intent clearer.
    Type method naming is automated to allow a bijection between type method
    names and target metaclass methods, as follows:
    * __meth__ in the metaclass corresponds to the __typemeth__ typemethod.
    * meth in the metaclass corresponds to the _typemeth typemethod.
    * _meth in the metaclass corresponds to the _type_meth typemethod.
    Typemethod accepts an optional string argument that names the target
    metaclass method (e.g. @typemethod(target='__getitem__')), in which case
    the decorated method can have any name its programmer likes. The type will
    still make an assignment to the conventional name, which then becomes an
    alias to the decorated typemethod. However this allows more flexibility in
    naming methods.
    """

    def __new__(cls, func=None, *, target=None):
        if func is None:
            return partial(cls, target=target)
        return classmethod.__new__(cls, func)

    def __init__(self, func=None, *, target=None):
        nxdescriptor.__init__(self)
        super().__init__(func)
        if target is None:
            try:
                self.target = self._target(func)
            except Exception as e:
                msg = "Typemethods must follow naming convention or " + \
                      "explicitly target a metaclass method with the " + \
                      "keyword 'target' in the decorator."
                raise ValueError(msg) from e
        else:
            self.target = target

    @classmethod
    def _target(cls, func):
        """Retrieves the metaclass target name from a function's name."""
        fname = func.__name__
        if fname.startswith('__type'):
            return fname.replace('__type', '__')
        elif fname.startwith('_type'):
            return fname.replace('_type', '')
        else:
            raise ValueError

    @classmethod
    def _func(cls, target):
        """Retrieves the conventional function name from a target."""
        if target.startswith('__'):
            return '__type' + target[2:]
        else:
            return '_type' + target

    def __metainit__(self, cls):
        if not hasattr(type(cls), self.target):
            msg = "{0} has invalid metaclass method target."
            raise MetaError(msg.format(self))
        setattr(cls, self._func(self.target), self)

#==============================================================================
### Slots property functions
#==============================================================================


def get_slots(obj):
    """Returns the slots of a slotted object."""
    return tuple(getattr(obj, s) for s in obj.__slots__)


def set_slots(obj, args):
    """Fills the slots of a slotted object."""
    list(setattr(obj, s, a) for s, a in zip(obj.__slots__, args))


def del_slots(obj):
    """Sets slots to None."""
    list(setattr(obj, s, None) for s in obj.__slots__)

#==============================================================================
### nxmeta
#==============================================================================


class nxmeta(nxtype):
    """The base type interpreter.

    nxmeta is not only the base class for NxType, which provides a base
    to library and application types, but also its metaclass, such that
    concrete types are instances of nxmeta.
    Unlike type, nxmeta accepts keyword arguments:
    * slots: a tuple of strings that is passed as __slots__ to the
    generated type;
    * Other keyword arguments are interpreted as type attributes, i.e.
    attributes used to generate different versions of types, and are assigned
    accordingly in the type that is created. Note that this interface
    prohibits named arguments 'bases', and 'slots' from being used as
    type attributes.
    """
    __typecount__ = 0  # A class variable that keeps track of instance count.

    @classmethod
    def __baptize__(mcl, bases, slots=None, **tpattrs):
        """Returns a programmatic type name.

        This is a default implementation meant to be overwritten in
        specialized metaclasses.
        """
        return mcl.__name__ + '_' + str(mcl.__typecount__ + 1)

    @classmethod
    def __prepare__(mcl, name, bases, slots=None, **tpattrs):
        return OrderedDict()

    def __new__(mcl, name, bases, namespace, slots=None, **tpattrs):
        if slots is not None:
            namespace['__slots__'] = slots
        reg = namespace['__nxdescriptors__'] = DescriptorRegistry()
        for k, v in namespace.items():
            if isinstance(v, nxdescriptor):
                v.name = k
                reg.register(v, type(v))
        return type.__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace, slots=None, **tpattrs):
        type.__init__(cls, name, bases, namespace)
        cls.__module__ = getmodule(sys._getframe(1))

        # New descriptor initialization
        for v in cls.__nxdescriptors__.values():
            v.__metainit__(cls)

        # Initializes type properties -always False at init
        # Property setting for types that are generated from metatypes
        # is handled by __new__.
        cls._is_archetype = False

        # Adds the slots property if applicable
        if hasattr(cls, '__slots__'):
            cls.slots = property(get_slots, set_slots, del_slots)

        # Keeps count of type creation
        if issubclass(cls, type):
            cls.__typecount__ = 0
        else:
            type(cls).__typecount__ += 1

    @property
    def is_metatype(cls):
        """True if a type has the ability to generate other types."""
        return cls.is_basetype or cls.is_archetype

    @property
    def is_basetype(cls):
        """True if a type can be extended programatically."""
        return cls._is_basetype

    def __getitem__(cls, key):
        """Enables bracket calls on types."""
        return cls.__typegetitem__(key)

    def __typegetitem__(cls, key):
        """Default type __typegetitem__ implementation."""
        return NotImplemented


#==============================================================================
### Phenotype metaclass
#==============================================================================


class phenotype(nxmeta):
    """Phenotypes are mixin types created from data descriptors.

    More specifically, patches support the type creation method called
    'overtyping', which consists in supplementing a base type with additional
    attributes and methods. The net result is equivalent to composing a
    new class from a base class and a mixin class, however overtyping
    provides a programmatic way to do this without having to explicitly
    declare the mixin class.
    # TODO: Revisit hash key. If two descriptors evaluate equal, then
    wouldn't a hash of the items sequence be the same?
    Moreover, overtyping allows type caching. So if a program needs to
    create an overtype of a given basetype by patching it with certain
    attributes, the basetype will first look up in a cache whether such
    an overtype has already been created, and fetch it accordingly. The
    cache is indexed by the content of the patches that are applied to
    create ovetypes. Specifically, Patch is hashable, and its hash key
    is based on the keys and *value types* of the mapping it exposes.
    This has a bit of a limitation, namely the fact that two patches that use
    the same descriptor and method names, and whose descriptors are of the
    same type, have the same hash key. However the thinking is that at this
    level of similarity the types should probably the same, and if not
    then traditional subclassing methods are always available.
    Under the hood, Patch implements an OrderedDict so that the order of
    descriptors can be preserved if needs be. The __init__ method can take
    sequential or keyword arguments, but obviously the former is required
    for the order in the OrderedDict to have any validity. The classmethod
    from_od is a helper to pass an OrderedDict directly.
    """

    def __init__(self, *items, **kwargs):
        self._dict = OrderedDict()
        for key, val in items:
            self._dict[key] = val
        for key, val in kwargs.items():
            self._dict[key] = val

    def __getitem__(self, key):
        return self._dict.__getitem__(key)

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return self._dict.__len__()

    @classmethod
    def from_od(cls, ordered_dict):
        patch = cls()
        patch._dict = ordered_dict

    def __hash__(self):
        """Hash key based on _dict keys and the types of the values."""
        tmap = {k: type(v) for k, v in self._dict.items()}
        # Need to sort in case items were input in different orders
        h = hash(tuple(sorted(tmap.items())))
        # Modifies the natural hash to always yield a positive value
        return h % ((sys.maxsize + 1) * 2)

##==============================================================================
## Assembler class
##==============================================================================
#
#
#class Assembler(nxobject):
#    """Assemblers are callables that return types.
#
#    The Assembler class as a whole is a type factory. Any Assembler
#    instance can be made to return any kind of classes, similarly to
#    the new_class function defined in the types standard library.
#    Assembler combines this functionality with a behavior akin to
#    partial in the functools standard library, such that Assembler
#    instances can be built recursively from multiple calls.
#    Assembler can automtically name a class if the metaclass used to
#    instantiate it features a classmethod __baptize__, which takes
#    bases, patch and **keys as arguments and returns a name.
#    """
#    # Class defaults, which may be overriden by subclasses.
#    bases = ()  # type bases.
#    metaclass = None  # metaclass used to create types.
#    patch = None  # optional dictionary to patch onto the namespace.
#    slots = None  # Optional tuple of __slot__ strings.
#    keys = None  # optional type keys, passed to the metaclass.
#
#    def __init__(self, *bases, metaclass=None, patch=None, slots=None, **keys):
#        """Initializes an Assembler. All arguments are optional.
#
#        :param *bases: a sequence of base classes.
#        :param metaclass: an optional metaclass.
#        :param patch: a dictionary of additional class attributes & methods.
#        :param slots: a tuple of __slots__ strings.
#        :param **keys: type keys / metaclass keyword arguments, as applicable.
#        :returns: an Assembler instance.
#        """
#        self.keys = {}
#        kwargs = ChainMap(keys, lmap('metaclass', 'patch', 'slots'))
#        self.update(*bases, **kwargs)
#
#    def update(self, *bases, metaclass=None, patch=None, slots=None, **keys):
#        """Updates an Assembler instance with new / additional parameters.
#
#        Note that all of the parameters except keys update through overwriting.
#        One cannot, say, add a base class or augment an existing patch with an
#        additional method. For this, a new bases tuple or patch dictionary
#        need to be created then passed to the update method.
#        Keys work the other way round, with a dictionary update.
#        """
#        if bases:
#            self.bases = bases
#        if metaclass is not None:
#            self.metaclass = metaclass
#        if patch is not None:
#            self.patch = patch
#        if slots is not None:
#            self.slots = slots
#        self.keys.update(keys)
#
#    def __call__(self, name=None, **kwargs):
#        """The main interface, returns either a type or an updated Assembler.
#
#        Admissible kwargs include bases, metaclass, patch and slots, which have
#        the result of updating those instance variables locally to the call.
#        Additional kwargs are treated as type keys updates.
#        If the attempt to create a class from the supplied argument
#        fails with a TypeError, an Assembler instance  with updated arguments
#        is returned in lieu of a type.
#        """
#        bases = kwargs.pop('bases', self.bases)
#        metaclass = kwargs.pop('metaclass', self.metaclass)
#        patch = kwargs.pop('patch', self.patch)
#        slots = kwargs.pop('slots', self.slots)
#        keys = self.keys.copy()
#        if patch is not None:
#            keys['patch'] = patch
#        if slots is not None:
#            keys['slots'] = slots
#        keys.update(kwargs)
#        metarg = {'metaclass': metaclass} if metaclass else {}
#        kwds = ChainMap(metarg, keys)
#        # Callable that returns function's return value if not a type.
#        assembler = lambda: type(self)(*bases, **kwds)
#        if not name:
#            # Find the appropriate metaclass by calling prepare with mock name
#            metaclass, _, _ = types.prepare_class('mock', bases, kwds)
#            if not hasattr(metaclass, '__baptize__'):
#                return assembler()
#            name = metaclass.__baptize__(bases, **keys)
#        try:
#            cls = types.new_class(name, bases, kwds)
#        # TODO: replace with prototype error
#        except TypeError:
#            return assembler()
#        cls.__module__ = getmodule(sys._getframe(1))
#        return cls

#==============================================================================
### Type properties
#==============================================================================



class metamethod(classmethod, nxdescriptor):
    """A classmethod closure that returns concrete instance methods.

    The metamethod decorator is used in the definition of archetypes to
    signal closure methods that return concrete implementation of methods in
    subtypes.
    """
    pass
