#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines descriptors used throughout the anaximander framework.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
from collections import OrderedDict, ChainMap
from functools import wraps
from inspect import signature
from itertools import count

from anaximander2.utilities import xprops, functions as fun
from anaximander2.utilities.cmpmixin import ComparableMixin

from . import NxMetaError

__all__ = ['NxDescriptor', 'ObjectDescriptor', 'TypeDescriptor',
           'MetaDescriptor', 'ObjectCharacter', 'TypeCharacter',
           'ObjectTypeProperty', 'MetaCharacter']

# =============================================================================
# Base classes
# =============================================================================


class NxDescriptor(ComparableMixin):
    """A basic data descriptor with registration mechanism.

    The descriptor accepts a one-time set name and class reference.
    The intended usage pattern is that the descriptor is declared
    in a class' namespace, and a metaclass collects the descriptors
    and assign them their name and the class in which they were
    declared. However it is possible to set those attributes at
    instantiation.
    The class reference is primarily useful for tracing and debugging
    purposes.
    NxDescriptor also features a registration mechanism, as practice
    demonstrates that it is often useful to enumerate the descriptors
    of a given type, whether handling an instance or a class.
    NxDescriptor is specialized into 3 abstract base classes to address
    3 layers of programming: objects, types, and metatypes.
    Each base class keeps a counter that is used to assign a creation id to
    descriptors and sort them by order of appearance.
    This is not necessary in Python 3.6+ when using metaclasses, because class
    declarations are now collected ordered. However there are cases where no
    metaclass may be used and the creation id can be used instead.
    Accordingly, NxDescriptors implement rich comparisons based on the
    descritptor id.
    """
    __cmpattrs__ = ('descriptor_id',)

    def __init__(self, name=None, cls=None):
        if name is not None:
            self._name = name
        if cls is not None:
            self._cls = cls
        try:
            self.descriptor_id = next(self.__counter__)
        except (TypeError, AttributeError):
            msg = "Cannot instantiate NxDescriptor without a counter."
            raise TypeError(msg)

    @xprops.cachedproperty
    def name(self):
        return None

    @xprops.cachedproperty
    def cls(self):
        return None

    def bind(self, name, cls):
        """Binds a descriptor to its declaring class and assigns its name."""
        self._name = name
        self._cls = cls

    def register(self, cls, registry_name='__nxdescriptors__'):
        """Registers self with the supplied class.

        Params:
            cls: A class in which to register the descriptor.
            registry_name: The attribute name of the class registry.
        """
        try:
            registry = getattr(cls, registry_name)
            registry[self.name] = self
        except (AttributeError, TypeError):
            msg = "NxDescriptors registration requires a valid class registry."
            raise NxMetaError(msg)

    def copy(self):
        copy_ = type(self)
        attrs = self.__dict__.copy()
        copy_.__dict__ = attrs
        return copy_

    @classmethod
    def collect(cls, bases, namespace, registry_name='__nxdescriptors__'):
        """Collects nxdescriptors from supplied namespace and bases.

        Params:
            cls: the NxDescriptor type whose instances are to be collected
            bases: a tuple of base classes
            namespace: a mapping of attributes
            registry_name: the expected attribute name of descriptor
            registries in the base classes
        Returns:
            an OrderedDict of attribute name to NxDescriptor instances

        The algorithm seeks the registry name on the supplied bases, or
        otherwise ignores them. It assumes that the registries are
        ordered dictionary that enumerate items in an ordered fashion.
        Generally, the namespace's descriptors are appended to the
        base class' descriptors. If there are mixin classes, i.e. any base
        beyond the first one, their descriptors are appended to the
        namespace's descriptors. Hence the descriptor order is as follows:
            bases[0] > namespace > bases[1:]
        In the case of the mixins, the order of the bases dictates the order
        of the descriptors, that is, the last mixin's descriptors are
        appended last.
        However it is possible for the namespace to redefine the order of
        the descriptors by explicitly restating them. In that case, the
        following rules are enforced:
            * if the namespace redefines descriptors that are in a base
            class, it must redefine all of them -this is to ensure a
            consistent order can be determined;
            * while it is possible to interlace the order of descriptors
            that were defined in base classes, the new order for the
            descriptors defined in any single class must be the same
            as in that base class.
        For instance:

            class C0:
                a = descriptor()
                b = descriptor()

            class C1:
                c = descriptor()
                d = descriptor()

            class C(C0, C1):  # legal
                a = descriptor()
                c = descriptor()
                b = descriptor()
                d = descriptor()

            class C(C0, C1):  # illegal, violates C1's order
                a = descriptor()
                b = descriptor()
                d = descriptor()
                c = descriptor()

        Note that it is up to the programmer to ensure that the restatements
        are consistne with the descriptors declared in the base classes. In
        other words, while the collect function enforces consistent rules
        in the naming and sequence of descriptors through inheritance,
        the definition of the descriptors themselves is not enforced.
        """
        # Collects descriptors from namespace
        ns_descriptors = fun.vfilter(fun.typecheck(cls), namespace)
        # Orders them by descriptor id
        ns_descriptors = OrderedDict(sorted(ns_descriptors,
                                            key=lambda i: i[1].descriptor_id))
        if not bases:
            return ns_descriptors
        # Fetches registries from the base
        base_descriptors = OrderedDict(getattr(bases[0], registry_name, {}))
        # Either namespace redefines them, or it appends them
        ns_descriptors_set = set(ns_descriptors)
        base_descriptors_set = set(base_descriptors)
        if ns_descriptors_set & base_descriptors_set:
            if not base_descriptors_set.issubset(ns_descriptors_set):
                msg = "A child class must redefine all or none of the " + \
                      "descriptors from its base class."
                raise NxMetaError(msg)
            # Test for compatible ordering
            try:
                fun.merge(ns_descriptors, base_descriptors)
            except ValueError:
                msg = "Descriptor ordering inconsistent with base class."
                raise NxMetaError(msg)
            descriptors = ns_descriptors
        else:
            descriptors = base_descriptors
            for k, v in ns_descriptors.items():
                descriptors[k] = v
        # Now we add the mixin descriptors
        mixin_descriptors = [OrderedDict(getattr(b, registry_name, {}))
                             for b in bases[1:]]
        # First we determine the key order, checking for duplication and
        # consistency
        try:
            keys = fun.merge(descriptors.keys(),
                             *[d.keys() for d in mixin_descriptors])
        except ValueError:
            msg = "Descriptor ordering inconsistent with base class."
            raise NxMetaError(msg)
        # Then we extract the values by building a chainmap
        dchain = ChainMap(descriptors, *mixin_descriptors)
        return OrderedDict((k, dchain[k]) for k in keys)

    @abc.abstractmethod
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return None

    def __set__(self, obj, value):
        raise AttributeError("Can't set attribute.")

    def __delete__(self, obj):
        raise AttributeError("Can't delete attribute.")

    def __repr__(self):
        return fun.iformat('name')(self)


class ObjectDescriptor(NxDescriptor):
    """Instance descriptor base class for regular types."""
    __counter__ = count()


# Ensures ObjectDescriptors are comparable
ObjectDescriptor.__cmptypes__ = (ObjectDescriptor,)


class TypeDescriptor(NxDescriptor):
    """Type descriptor base class for metaclasses."""
    __counter__ = count()

    def set_typical_property(self, cls):
        """Implements an ObjectTypeProperty on the supplied class.

        The intended pattern is to make a type descriptor available to
        that type's instances as read-only property.
        """
        prop = ObjectTypeProperty(self.name)
        cls.__dict__[self.name] = prop

# Ensures TypeDescriptors are comparable
TypeDescriptor.__cmptypes__ = (TypeDescriptor,)


class MetaDescriptor(NxDescriptor):
    """Descriptor declared in an archetype but aimed at the metaclass."""
    __counter__ = count()
    __typedescriptor__ = None  # The target TypeDescriptor subclass

    def transfer(self, mcl):
        """Transfers self's declarations to a type descriptor."""
        typedescriptor = self.__typedescriptor__()
        attrs = self.__dict__.copy()
        del attrs['descriptor_id']
        typedescriptor.__dict__ = attrs
        setattr(mcl, self.name, typedescriptor)

# Ensures MetaDescriptors are comparable
MetaDescriptor.__cmptypes__ = (MetaDescriptor,)


# =============================================================================
# Abstract implementations of typical patterns
# =============================================================================


class ProtectedAttribute(NxDescriptor):
    """A descriptor that reads a cached attribute.

    This class also offers enhanced attribute functionalities, i.e.
    default and validation.
    The validate function can be supplied either in simple form, as a callable
    that takes a single value argument and returns True or False, or a
    more complete function that takes as arguments:
        * The object whose attribute is set;
        * The attribute instance itself;
        * The value that is being set.
    Either way, the validate attribute of ProtectedAttribute has the more
    complex signature -if the simple form is supplied, it is decorated
    to that end.
    The validate function should return a boolean and the ProtectedAttrbute
    setter will raise a ValueError if False is returned. However validate
    may also raise its own exception, in which case the exception type
    is used by the setter.
    It is also possible to define a class-level method by overwriting
    __validate__. The method and the instance-defined function are
    cumulative.
    """

    def __init__(self, default=None, validate=None, name=None, cls=None,
                 cache=None):
        super().__init__(name, cls)
        self.default = default
        self.validate = validate
        self._cache = cache

    def bind(self, name, cls):
        super().bind(name, cls)
        if self._cache is None:
            self._cache = '_' + self.name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # We prefer obj.__dict__ to getattr, as this enables cachedproperty
        # to operate on instances or types. With the latter, getattr
        # would look up the inheritance chain which is not what we want.
        try:
            return obj.__dict__[self._cache]
        except KeyError:
            return self.default

    def __set__(self, obj, value):
        try:
            assert self.validate(obj, self, value)
        except Exception as e:
            msg = f"Invalid value {value} passed to attribute " + \
                  f"{self.name} of {obj}"
            etype = type(e)
            if etype is AssertionError:
                etype = ValueError
            raise etype(msg)
        setattr(obj, self._cache, self.default)

    def __delete__(self, obj):
        try:
            delattr(obj, self._cache)
        except AttributeError:
            pass

    def __validate__(self, obj, value):
        """A validation method to be overwritten in subclasses."""
        return True

    @property
    def validate(self):
        if hasattr(self, '_validate'):
            return self._validate
        return None

    @validate.setter
    def validate(self, func):
        if func is None:
            def validator(obj, attr, value):
                return self.__validate__(obj, value)
        else:
            sig = signature(func)
            # Simple validate function is possible but its signature is changed

            @wraps(func)
            def validator(obj, attr, value):
                class_valid = self.__validate__(obj, value)
                if len(sig.parameters) is 1:
                    return func(value) and class_valid
                else:
                    return func(obj, attr, value) and class_valid
        setattr(self, '_validate', validator)

    @classmethod
    def reset(cls, obj):
        """Resets attributes on supplied object."""
        objtype = type(obj)
        for attr in dir(objtype):
            v = getattr(objtype, attr, None)
            if isinstance(v, cls):
                v.__delete__(obj)

    def validator(self, func):
        """A decorator to declare a validation function externally.

        The expected signature of func is (obj, attr, value), where
        obj will usually be input as self in the declararing class, and
        refers to the instance on which the attribute (attr) is being
        set with value.
        """
        self.validate = func
        return self


class SetOnceAttribute(ProtectedAttribute):
    """A ProtectedAttribute that must be set once and only once.

    Defaults are not permitted on SetOnceAttributes.
    """

    def __init__(self, validate=None, name=None, cls=None, cache=None):
        super().__init__(None, validate, name, cls, cache)

    def __set__(self, obj, value):
        if self._cache not in obj.__dict__:
            super().__set__(obj, value)
        else:
            raise AttributeError("Can't set attribute.")

# =============================================================================
# Concrete types
# =============================================================================


class ObjectCharacter(SetOnceAttribute, ObjectDescriptor):
    pass


class TypeCharacter(SetOnceAttribute, TypeDescriptor):
    """TypeCharacters are designed for parameters of parametric types."""
    pass


class ObjectTypeProperty(ObjectDescriptor):
    """A descriptor that reads a property of the type.

    The name is admittedly a bit confusing, but it makes sense in light of
    the definition: these are instance-level properties that make
    accessible a type descriptor as a read-only property. Because type
    descriptors exist in the metaclass' dictionary, they cannot be
    readily accessed by instances. In some cases that's fine, but
    TypeCharacters in particular should be accessible to instances
    transitively, in the same way that a class variable is accessible
    to instances.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(type(obj), self.name)


class MetaCharacter(MetaDescriptor):
    """A metadescriptor for archetypical characters."""
    __typedescriptor__ = TypeCharacter
