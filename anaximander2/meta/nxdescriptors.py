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
from typing import get_type_hints

from anaximander2.utilities import xprops, functions as fun

from . import NxMetaError

__all__ = ['NxDescriptor', 'MetaDescriptor', 'NxAttribute',
           'TypeAttribute', 'TypeParameter', 'TypeAttributeProperty',
           'newtypemethod', 'typeinitmethod', 'typeproperty']

# =============================================================================
# Base classes
# =============================================================================


class Registrable:
    """A class attribute that implements a registration mechanisms.

    Metaclasses can collect and name registrables, which provide a base class
    to descriptors, metadescriptors, and other class attributes that
    form collections.
    Registrables accept a name and class reference.
    The intended usage pattern is that registrables are declared
    in a class' namespace, and a metaclass collects them
    and assign them their name and the class in which they were
    declared. However it is possible to set those attributes at
    instantiation.
    Registrables also takes an optional type attribute. It can be
    assigned directly through __init__, or it will be looked up in
    the declaring class' annotations in the bind method.
    The class reference is primarily useful for tracing and debugging
    purposes.
    Registables feature a registration mechanism, as practice
    demonstrates that it is often useful to enumerate the descriptors
    of a given type, whether handling an instance or a class.
    Registrable itself is an abstract base class, but subclasses can setup
    their own counter and registry name. The counter is used to assign a
    creation id to registrables and sort them by order of appearance in the
    code. This is not necessary in Python 3.6+ when using metaclasses, because
    class declarations are collected ordered. However there are cases where no
    metaclass may be involved and the creation id can be used instead.
    Accordingly, Registrable implements rich comparisons based on the
    descritptor id.
    """
    # Placeholder registry name for types that declare concrete registrables
    # In concrete subtypes, __counter__ needs to be a counter object
    __registry__ = None
    __counter__ = None

    def __init__(self, name=None, cls=None):
        if name is not None:
            self._name = name
        if cls is not None:
            self._cls = cls
        try:
            self.registration_id = next(self.__counter__)
        except (TypeError, AttributeError):
            msg = "Cannot instantiate Registrable without a counter."
            raise TypeError(msg)

    @xprops.cachedproperty
    def name(self):
        return None

    @xprops.cachedproperty
    def cls(self):
        return None

    def bind(self, name, cls):
        """Binds a registrable to its declaring class and assigns its name."""
        self._name = name
        self._cls = cls

    def register(self, cls):
        """Adds self to the pertinent registry in cls."""
        try:
            registry = getattr(cls, self.__registry__)
        except AttributeError:
            registry = OrderedDict()
            setattr(cls, self.__registry__, registry)
        registry[self.name] = self

    @classmethod
    def retrieve(cls, obj, *types):
        """Retrieves registrables registered on object, with type filtering.

        GOTCHA: subclasses can only be retrieved if they are registered in
        the same registry as cls.
        """
        if not types:
            types = (cls,)
        try:
            registry = getattr(obj, cls.__registry__)
        except AttributeError:
            return OrderedDict()
        registrables = fun.vfilter(fun.typecheck(*types), registry)
        return OrderedDict(registrables)

    def copy(self):
        copy_ = type(self)
        attrs = self.__dict__.copy()
        copy_.__dict__ = attrs
        return copy_

    @classmethod
    def collect(cls, klass, namespace):
        """Binds, collects and registers registrables on newly created class.

        Params:
            cls: the registrable type whose instances are to be collected
            klass: the class on which registrables are collected
            namespace: the namespace declared by klass
        Returns:
            an OrderedDict of attribute name to registrable instances, which
            is also assigned to klass with the name cls.__registry__.

        The algorithm seeks the registry name on the supplied bases, or
        otherwise ignores them. It assumes that the registries are
        ordered dictionaries that enumerate items in an ordered fashion.
        Generally, the namespace's registrables are appended to the
        base class' registrables. If there are mixin classes, i.e. any base
        beyond the first one, their registrables are appended to the
        namespace's registables. Hence the registration order is as follows:
            bases[0] > namespace > bases[1:]
        In the case of the mixins, the order of the bases dictates the order
        of the registrables, that is, the last mixin's registrables are
        appended last.
        However it is possible for the namespace to redefine the order of
        the registrables by explicitly restating them. In that case, the
        following rules are enforced:
            * if the namespace redefines registrables that are in the base
            class, it must redefine all of them -this is to ensure a
            consistent order can be determined;
            * while it is possible to interlace the order of registrables
            that were defined in base classes, the new order for the
            registrables defined in any single class must be the same
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
        are consistne with the registrables declared in the base classes. In
        other words, while the collect function enforces consistent rules
        in the naming and sequence of descriptors through inheritance,
        the definition of the registrables themselves is not enforced.
        """
        # Collects registrables from namespace
        ns_rgs = fun.vfilter(fun.typecheck(cls), namespace)
        # Orders them by registration id
        ns_rgs = OrderedDict(sorted(ns_rgs,
                                    key=lambda i: i[1].registration_id))
        for name, rg in ns_rgs.items():
            rg.bind(name, klass)
        bases = klass.__bases__
        if not bases:
            return ns_rgs
        # Fetches registries from the base
        base_rgs = OrderedDict(getattr(bases[0], cls.__registry__, {}))
        # Either namespace redefines them, or it appends them
        ns_rgs_set = set(ns_rgs)
        base_rgs_set = set(base_rgs)
        if ns_rgs_set & base_rgs_set:
            if not base_rgs_set.issubset(ns_rgs_set):
                msg = "A child class must redefine all or none of the " + \
                      "registrable attributes from its base class."
                raise NxMetaError(msg)
            # Test for compatible ordering
            try:
                fun.merge(ns_rgs, base_rgs)
            except ValueError:
                msg = "Registrables ordering inconsistent with base class."
                raise NxMetaError(msg)
            registrables = ns_rgs
        else:
            registrables = base_rgs
            for k, v in ns_rgs.items():
                registrables[k] = v
        # Now we add the mixin registrables
        mixin_rgs = [OrderedDict(getattr(b, cls.__registry__, {}))
                     for b in bases[1:]]
        # First we determine the key order, checking for duplication and
        # consistency
        try:
            keys = fun.merge(registrables.keys(),
                             *[d.keys() for d in mixin_rgs])
        except ValueError:
            msg = "Registables ordering inconsistent with base class."
            raise NxMetaError(msg)
        # Then we extract the values by building a chainmap
        rgchain = ChainMap(registrables, *mixin_rgs)
        registry = OrderedDict((k, rgchain[k]) for k in keys)
        setattr(klass, cls.__registry__, registry)
        return registry

    def __repr__(self):
        return fun.iformat('name')(self)


class NxDescriptor(Registrable):
    """A basic data descriptor with registration mechanism."""
    __registry__ = '__nxdescriptors__'
    __counter__ = count()

    @abc.abstractmethod
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return None

    def __set__(self, obj, value):
        raise AttributeError("Can't set attribute.")

    def __delete__(self, obj):
        raise AttributeError("Can't delete attribute.")


class MetaDescriptor(Registrable):
    """A declaration that can be turned into a descriptor by a metaclass.

    The main funcion of metadescriptors is to simplify metaprogramming by
    declaring properties and methods applicable to types in a regular
    class declaration. So-called archetypes are abstract base types
    that generate their own metatype. Archetype declarations can include
    metadescriptors, which may then be turned into descriptors of that
    metatype. MetaDescriptor is subclassed into specialized types that
    provide the intent and implementation for different such behaviors.
    The class acts as a descriptor factory through its __call__ method,
    returning a regular descriptor when called with a cls attribute.
    The assign method assigns the descriptor to the cls as a side effect.
    One pattern is to wrap a descriptor as an attribute of a metadescriptor.
    In that case, the virtue of the metadescriptor is strictly to provide
    a routing mechanism for declarative class statements.
    Another pattern is to implement an interface and subclass it into
    a regular descriptor and a metadescriptor.
    """
    __registry__ = '__metadescriptors__'
    __counter__ = count()

    def __repr__(self):
        return fun.iformat('name')(self)

    @abc.abstractmethod
    def __call__(self, mcl):
        return None

    def assign(self, mcl):
        """Generates a descriptor and assigns it to mcl."""
        descriptor = self(mcl)
        setattr(mcl, self.name, descriptor)
        if isinstance(descriptor, NxDescriptor):
            descriptor.register(mcl)

# =============================================================================
# Abstract implementations of typical patterns
# =============================================================================


class Caller:
    """A wrapper for setting callable defaults in NxAttributes.

    Caller accepts functions with either no argument, or a single argument
    which should be the object on which a descriptor is being set.
    """

    def __init__(self, func, validate=False):
        self.func = func
        self.validate = validate

    def __call__(self, obj):
        return self.func(obj)

    @property
    def func(self):
        if hasattr(self, '_func'):
            return self._func
        return None

    @func.setter
    def func(self, f):
        """Sets the function and a standard signature."""
        try:
            sig = signature(f)
        except ValueError:
            # We assume f is a built-in type such as list
            params = 0
        else:
            if len(sig.parameters) is 0:
                params = 0
            elif len(sig.parameters) is 1:
                params = 1
            else:
                msg = "Default caller can take either 0 or 1 argument."
                raise ValueError(msg)

        @wraps(f)
        def caller(obj):
            if params is 0:
                return f()
            else:
                return f(obj)

        setattr(self, '_func', caller)


def call(func, validate=False):
    """A helper to define callable default values in NxAttributes."""
    return Caller(func, validate)


class NxAttributeInterface(Registrable):
    """A specification interface for descriptors that read a cached attribute.

    NxAttribute is subclassed into both an NxDescriptor and a MetaDescriptor
    to enable type attributes.
    This class offers enhanced attribute functionalities, i.e.
    default and validation.

    The default can optionally be generated by a callable if wrap under
    the call function. The wrapped callable can either take no argument or
    one argument which is expected to be the object upon which the attribute
    is set at runtime. Hence possible declarations include the following,
    which are all equivalent:
        default=call(lambda obj: list())
        default=call(lambda: list())
        default=call(list)
    call takes an optional validate argument set as True or False (default),
    which determines whether default values generated by a caller undergo
    validation.

    The validate function can be supplied either in simple form, as a callable
    that takes a single value argument and returns True or False, or a
    more complete function that takes as arguments:
        * The object whose attribute is set;
        * The attribute instance itself;
        * The value that is being set.
    Either way, the validate attribute of NxAttribute has the more
    complex signature -if the simple form is supplied, it is decorated
    to that end.
    The validate function should return a boolean and the NxAttrbute
    setter will raise a ValueError if False is returned. However validate
    may also raise its own exception, in which case the exception type
    is used by the setter.
    It is also possible to define a class-level method by overwriting
    __validate__. The method and the instance-defined function are
    cumulative.
    Additionally, if type has been defined either in the descriptor's
    constructor or through an annotation, type checking is enforced by
    automatic decoration of the validate function.
    This calls for the additional argument nullable, which determines
    whether None is an acceptable value (default is True).
    If nullable is True and a None value is supplied, the validate
    function is bypassed, so there is no need to specifically handle that
    case.

    If set_once is True, the attribute can only be set once for a given
    object -though it is technically possible to bypass this restriction
    by emptying the cache, which is enabled by the delete directive. Note
    that calling the attribute before it is set will have the effect of
    filling the cache.
    """

    def __init__(self, default=None, nullable=True, validate=None,
                 set_once=False, name=None, cls=None, type_=None, cache=None):
        """NxAttribute constructor.

        Attrs:
            default: a default value returned by the descriptor when no value
                has been set.
            nullable: whether it is acceptable to set a None value on the
                descriptor.
            validate: a callable that validates supplied values. Should
                either take a single value argument, or (obj, attr, value)
                where obj is the object whose attribute is being set,
                attr is the descriptor instance (i.e. self in the context
                of this documentation string), and value the supplied value.
            set_once: if True the attribute can no longer be set once a value
                has been cached.
            name: the descriptor's name.
            cls: the class that declares the descriptor.
            type_: the expected type_ for the descriptor.
            cache: an optional string for naming the attribute that holds
                the values that are set. It defaults to '_' + name and should
                rarely be manipulated.
        """
        super().__init__(name, cls)
        self.default = default
        self.nullable = nullable
        self.validate = validate
        self.validate_param = validate  # used to copy the descriptor
        self.set_once = set_once
        if type_ is not None:
            self._type = type_
        if cache is None and name is not None:
            self._cache = '_' + self.name
        else:
            self._cache = cache

    @xprops.cachedproperty
    def type(self):
        return object

    @property
    def cache(self):
        return self._cache

    def bind(self, name, cls):
        super().bind(name, cls)
        try:
            type_ = get_type_hints(cls)[name]
        except KeyError:
            pass
        else:
            self._type = type_
        if self._cache is None:
            self._cache = '_' + self.name

    def __validate__(self, obj, value):
        """A base validation method, can be specialized in subclasses."""
        if value is None and self.nullable:
            return True
        elif not isinstance(value, self.type):
            raise TypeError()
        else:
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
                if value is None:
                    return class_valid
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


class MetaMethod(MetaDescriptor):
    """A wrapper for methods aimed at a metaclass derived from an archetype."""

    def __init__(self, method, name=None, cls=None):
        self.method = method
        super().__init__(name, cls)

    def __call__(self, cls):
        return self.method


# =============================================================================
# Concrete types
# =============================================================================


class NxAttribute(NxAttributeInterface, NxDescriptor):
    """A descriptor that validates and caches a supplied value."""

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # We prefer obj.__dict__ to getattr, as this enables cachedproperty
        # to operate on instances or types. With the latter, getattr
        # would look up the inheritance chain which is not what we want.
        try:
            return obj.__dict__[self._cache]
        except KeyError:
            if isinstance(self.default, Caller):
                value = self.default(obj)
                if self.default.validate:
                    self.__set__(obj, value)
                    return value
            else:
                value = self.default
            setattr(obj, self._cache, value)
            return value

    def __set__(self, obj, value):
        if self.set_once and self._cache in obj.__dict__:
            raise AttributeError("Can't set attribute.")
        try:
            assert self.validate(obj, self, value)
        except Exception as e:
            msg = f"Invalid value {value} passed to attribute " + \
                  f"{self.name} of {obj}"
            etype = type(e)
            if etype is AssertionError:
                etype = ValueError
            raise etype(msg)
        setattr(obj, self._cache, value)

    def __delete__(self, obj):
        try:
            delattr(obj, self._cache)
        except AttributeError:
            pass


class NxMetaAttribute(NxAttributeInterface, MetaDescriptor):
    """A metadescriptor for nxattributes."""
    __dtype__ = NxAttribute

    def __call__(self, mcl):
        """Transfers self's declarations to a type descriptor."""
        descriptor = self.__dtype__()
        attrs = self.__dict__.copy()
        del attrs['registration_id']
        descriptor.__dict__ = attrs
        return descriptor


class TypeAttributeProperty(NxDescriptor):
    """A descriptor that reads an attribute of the type.

    These are instance-level properties that make accessible an attribute of
    the type as a read-only property. Because descriptors of types are
    part of a metaclass' dictionary, they cannot be readily accessed by
    instances. In some cases that's fine, but TypeAttributes in particular
    should generally be accessible to instances transitively, in the same way
    that a class variable is accessible to instances.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(type(obj), self.name)


class TypeAttribute(NxMetaAttribute):
    """A metadescriptor for setting type-level attributes."""

    def set_instance_property(self, cls):
        """Implements a TypeAttributeProperty on the supplied class.

        The intended pattern is to make a type descriptor available to
        that type's instances as read-only property.
        """
        prop = TypeAttributeProperty(self.name, cls)
        # Resets possible cached value on cls
        try:
            delattr(cls, self.cache)
        except AttributeError:
            pass
        setattr(cls, self.name, prop)
        prop.register(cls)

    def is_defined(self, cls):
        """True if the attribute has been set to a non-None value in cls."""
        definition = getattr(cls, self.name, None)
        if isinstance(definition, MetaDescriptor):
            return False
        return definition is not None

    # Property included to present the same interface as TypeParameter
    @property
    def covariant(self):
        return False


class TypeParameter(TypeAttribute):
    """A set_once, required type attribute.

    Type parameters must be set to a non-None value in order for a type to
    not be considered abstract. It is permitted  to create types that do not
    supply values to a type parameter but, these will be marked as abstract
    -supposing the default is None. If a default value is supplied, this
    property is moot.
    The set_once property insures that the parameter stays the same after
    type instantiation -so it is possible to run methods at instantiation
    that utilize type parameters while guaranteeing these parameters won't
    be changed.
    Immutability also makes type parameters candidates for type registration
    keys. If the flag key is set to True, types may be registered in their
    metaclass. The registration key is either set by a single type parameter,
    or if multiple type parameters have the key flag set to True, the type
    registration key is a tuple of these parameter values.
    The covariant_from argument can take a type argument. It dictates that
    subtypes from an archetype that declares the type parameter must be
    covariant with respect to that parameter. Note that covariance is meant
    here in terms of concrete inheritance, rather than a mere issubclass
    registration. As a result of passing a covariant_from argument, an
    archetype can only accept a single type parameter. In this case,
    a few behaviors are automatically enforced: default is set to the
    type supplied to the covariant_from argument, nullable is set to False,
    key is set to True, and type_ is the metaclass of the base parameter.
    As a result, an archetype that declares a covariant type parameter
    will automatically receive the base parameter.
    """

    def __init__(self, default=None, nullable=True, validate=None, key=True,
                 covariant_from=None, name=None, cls=None, type_=None,
                 cache=None):
        super().__init__(default, nullable, validate, True,
                         name, cls, type_, cache)
        self.key = key
        if covariant_from is not None:
            if not isinstance(covariant_from, type):
                msg = "Covariance requires a base type argument."
                raise TypeError(msg)
            self.default = covariant_from
            self.nullable = False
            try:
                assert covariant_from.is_archetype
            except (AttributeError, AssertionError):
                self._type = type(covariant_from)
            else:
                self._type = covariant_from.__metatype__
            self.key = True
        self.covariant_from = covariant_from

    @property
    def covariant(self):
        return self.covariant_from is not None

    def rebase(self, cls):
        """Rebases cls based on covariance setting."""
        try:
            assert self.covariant_from.is_archetype
        except (AttributeError, AssertionError):
            covariance_base = self.covariant_from
        else:
            covariance_base = self.covariant_from.__basetype__
        type_ = getattr(cls, self.name)
        # If type_ is covariant_from, no rebasing is necessary
        if type_ is self.covariant_from:
            return cls
        mro = type_.mro()
        if covariance_base not in mro:
            msg = "Improper type parameter setting."
            raise NxMetaError(msg)
        # First we test whether type_ itself is already registered
        if type_ in cls.archetype.registry:
            base_class = cls.archetype[type_]
        else:
            base_type = mro[1]
            if base_type is covariance_base:
                if base_type is not self.covariant_from:
                    # case where covariant_from is an archetype
                    base_type = self.covariant_from
            base_class = cls.archetype[base_type]
        if base_class is cls.archetype:
            base_class = base_class.__basetype__
        if issubclass(cls, base_class):
            return cls
        # If cls doesn't inherit from base, we create a new class that does
        kwargs = {self.name: type_}
        new = base_class.subtype(name=cls.__name__, overtype=cls.__overtype__,
                                 **kwargs)
        # If additional descriptors exist in cls, they are ported to new
        xattrs = set(cls.__dict__) - set(new.__dict__)
        for k in xattrs:
            setattr(new, k, getattr(cls, k))
        return new

    def assign(self, mcl):
        super().assign(mcl)
        if not self.covariant:
            return
        covariant_name = self.covariant_from.__name__.lower()
        method_name = 'rebase_from_' + covariant_name
        method = self.rebase
        typemethod = NewTypeMethod(method, method_name, mcl)
        typemethod.register(mcl)


class NewTypeMethod(MetaMethod):
    """A metamethod invoked at new type creation.

    The method must take a class as input and return a class as well.
    """
    pass


def newtypemethod(method):
    """A convenience decorator, equivalent to NewTypeMethod."""
    return NewTypeMethod(method)


class TypeInitMethod(MetaMethod):
    """A metamethod invoked at type initialization.

    The method must take a class as input. Return values are ignored.
    """
    pass


def typeinitmethod(method):
    """A convenience decorator, equivalent to TypeInitMethod."""
    return TypeInitMethod(method)


class TypeProperty(MetaDescriptor):
    """A property wrapper aimed at the metaclass."""

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, name=None,
                 cls=None):
        super().__init__(name, cls)
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __call__(self, cls):
        return property(self.fget, self.fset, self.__doc__)

    def getter(self, fget):
        self.fget = fget
        if fget is not None and self.__doc__ is None:
            self.__doc__ = fget.__doc__
        return self

    def setter(self, fset):
        self.fset = fset
        return self

    def deleter(self, fdel):
        self.fdel = fdel
        return self


def typeproperty(fget):
    """A convenience decorator, equivalent to TypeProperty."""
    return TypeProperty(fget)
