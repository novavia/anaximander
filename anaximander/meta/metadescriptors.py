#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metadescriptors enable metatype behaviors from archetype declarations.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
from collections import ChainMap, OrderedDict

from ..utilities import xprops, nxattr

__all__ = ['typeattribute', 'metacharacter', 'typeproperty',
           'cachedtypeproperty', 'typemethod', 'classtypemethod',
           'newtypemethod', 'typeinitmethod', 'metamethod', 'metainstance',
           'ValidationError', 'TypeDescriptor']

# =============================================================================
# Utilities
# =============================================================================


# Container for metadescriptor registries found in anaximander metaclasses
# 'metaregistries' maps MetaDescriptor types to a MetaRegistryFactory instance.
# Not all MetaDescriptor types require a dedicated registry, so this is
# implemented as an optional class decorator metaregistry.
# Every metaregistry entry creates an OrderedDict in anaximander metaclasses.
# However, two behaviors are possible: a metaregistry can accumulate entries
# through inheritance chains between metaclasses, or it can be reset to an
# empty OrderedDict in each metaclass. This option is set by adding
# reset=True in the registry decorator (False by default).
metaregistries = dict()


@nxattr.s
class MetaRegistryFactory:
    """A utility class that facilitates the creation of metaregistries.

    Note that metaregistries *do not* support multiple inheritance, which
    is in keeping with the fact that anaximander types themselves do not
    suppport multiple inheritance -they implement traits instead.
    """
    name = nxattr.ib()
    reset = nxattr.ib(default=False)

    def __call__(self, mcl):
        """Creates an OrderedDict, copying elements if necessary.

        The registry is also returned by the call.
        """
        try:
            parent = getattr(mcl, self.name)
        except AttributeError:
            registry = OrderedDict()
        else:
            if self.reset is True:
                registry = OrderedDict(parent)
            else:
                registry = OrderedDict()
        setattr(mcl, self.name, registry)
        return registry


def metaregistry(name, reset=False):
    """Instructs a MetaDescriptor class to add a metaregistry.

    params:
        name (str): The name given to the metaregistry in anaximander
            metaclasses.
        reset (bool): If True, the metaregistry is reset to an empty
            OrderedDict in each metaclass. Otherwise, metaregistry content
            is copied from the parent metaclass.
    """
    factory = MetaRegistryFactory(name, reset)

    def register(cls):
        """Registers the decorated class in metaregistries."""
        metaregistries[cls] = factory
        return cls

    return register

# =============================================================================
# Base metadescriptor
# =============================================================================


class MetaDescriptorError(Exception):
    """Customized error class for MetaDescriptor errors."""
    pass


class BindingError(MetaDescriptorError):
    """Exception invoked when metadescriptor binding fails."""
    pass


# Attributes for MetaDeclaration
metadeclaration_attrs = {'cls': nxattr.ib(init=False),
                         'name': nxattr.ib(init=False)}


@nxattr.s(these=metadeclaration_attrs, init=False)
class MetaDeclaration(abc.ABC):
    """MetaDeclaration are a generalization of metadescriptors.

    MetaDeclarations are collected by metaclasses and used to implement
    various initialization routines. The main application is MetaDescriptors.
    However another application is MetaInstances, which are cached
    instances that are created along with a type.
    """

    @xprops.singlesetproperty
    def cls(self):
        """The declaring class, set by NxType."""
        return None

    @xprops.singlesetproperty
    def name(self):
        """The declared name, set by NxType."""
        return None    


@metaregistry('__metadescriptors__')
class MetaDescriptor(MetaDeclaration):
    """Base class for metadescriptors.

    Metadescriptors are intended to be inserted in type declarations
    to modify behavior for derived types. Their scope is types rather than
    objects, and thus they are declarative devices that get processed to
    become descriptors in a metaclass, hence the name metadescriptor.
    The NxType base metaclass systematically collects metadescriptors found
    in type declarations, strips them from the type's namespace, and park
    them into a __metadeclarations__ dictionary. For regular types this
    accomplishes nothing immediately but the __metadeclarations__ are
    inherited and possibly updated by children classes. However if a type is
    decorated with @archetype then the __metadeclarations__ dictionary is
    interpreted in order to create a new metaclass that implements the
    behaviors programmed in the metadescriptors.
    Metadescriptor behavior is bound to a metaclass in two stages. In the
    first stage, NxType passes the declaring type to each metadescriptor.
    In the second stage, the __call__ method of the metadescriptor instance
    is called and supplied a metaclass whose dictionary gets modified as
    a result.
    """

    def register(self, mcl):
        """Registers self with the supplied metaclass."""
        for mdtype in type(self).__mro__:
            try:
                registry = mcl.metaregistries[mdtype]
            except KeyError:
                pass
            else:
                registry[self.name] = self

    def __call__(self, mcl):
        """Adds targeted behavior to the supplied metaclass."""
        if self.cls is None or self.name is None:
            msg = "Cannot call metadescriptor instance without class or name."
            raise BindingError(msg)
        self.register(mcl)

# =============================================================================
# Type attributes
# =============================================================================


class ValidationError(MetaDescriptorError, ValueError):
    """Raised if wrong value passed to a TypeAttribute."""
    pass


# Attributes for TypeAttribute
typeattribute_attrs = {'cls': nxattr.ib(init=False),
                       'name': nxattr.ib(init=False),
                       'default': nxattr.ib(default=None),
                       'validate': nxattr.ib(default=None)}


@metaregistry('__typeattributes__')
@nxattr.s(these=typeattribute_attrs, inherit=False)
class TypeAttribute(MetaDescriptor):
    """A metadescriptor that sets a type keyword argument.

    params:
        validate (func): a validation function that will run on
            values supplied to new types for the type attribute.
            Should simply return True upon success.
        default: a default value that is passed to new types in
            case no value is supplied. Defaults to None. The default can
            also be a callable that takes a type as its only argument. In
            that case, the value of the attribute for a type is computed
            dynamically upon first call.

    TypeAttributes are intended to be immutable: they are set at type
    creation and cannot be further modified.
    """
    # The reset token is used to reset the cached value of type attributes
    # whose default is a callable.
    _reset_token = object()

    def __call__(self, mcl):
        super().__call__(mcl)
        type_property = property(lambda t: getattr(t, '_' + self.name))
        setattr(mcl, self.name, type_property)
        inst_property = property(lambda i: getattr(type(i), self.name))
        try:
            setattr(self.cls, self.name, inst_property)
        # This happens when an archetype redefines a typeattribute
        # declared in a parent archetype.
        except AttributeError:
            pass

    def assign(self, class_or_namespace, value):
        """Assign value to cls or namespace.

        attrs:
            class_or_namespace: either a type or a dict-like object.
            value: the value to assign to the class or namespace.

        raises:
            ValidationError: if the value doesn't check self.validate
        """
        attr = '_' + self.name
        if self.validate is not None:
            try:
                assert self.validate(value)
            except AssertionError:
                if value is not self._reset_token:
                    raise ValidationError()
        if isinstance(class_or_namespace, type):
            setattr(class_or_namespace, attr, value)
        else:
            class_or_namespace[attr] = value

    @classmethod
    def update(cls, mcl, namespace, **kwargs):
        """Bulk assign to namespace from kwargs.

        attrs:
            mcl: a metaclass holding a __typeattributes__ dictionary.
            namespace: a dict-like object to be supplied to a new type.
            kwargs: a mapping containing type attribute assignments.

        raises:
            ValidationError: if at least one assignment fails per assign.

        If namespace contains typeattribute keywords, these take
        precedence over kwargs. This ensures that declarations made in
        a type have priority over kwargs passed to the metaclass at class
        instantiation. This order of priority is consistent with inheritance
        rules, i.e. descriptors in a class declaration overwrite those found
        in the class' bases that are passed to its metaclass.
        The method also cleans up the class_or_namespace of such declarations,
        so that the type property won't get overwritten.
        """
        chainmap = ChainMap(namespace, kwargs)
        for k, v in mcl.__typeattributes__.items():
            try:
                value = chainmap.pop(k, chainmap[k])
            except KeyError:
                # If the TypeAttribute is callable, we set the cache to
                # the reset token so the new type has a chance to define
                # its own value.
                if callable(v.default):
                    v.assign(namespace, cls._reset_token)
                pass
            else:
                v.assign(namespace, value)

    def _reset(self, type_):
        """Used to reset type cache to override inheritance.

        If self has a callable default and a new type inherits a cached value
        from its base, the assignment made in the base or one of its
        superclass persists --unless the cached value is the default
        value of the base in which case the default is run on the new type.
        Note that this leaves open a corner case in which there was a
        manual assignment of a TypeAttribute that happens to be the
        default value of one of the base types. In that case, the value
        will be overriden even though it was not the intention. Solving
        this corner case would require additional flags which creates
        complexity for a seemingly extremely marginal occurrence. But this
        note is here as a warning.
        """
        attr = '_' + self.name
        try:
            assignment = getattr(type_, attr)
            if assignment is not self._reset_token:
                setattr(type_, attr, assignment)
                return
        except AttributeError:
            # No preceding value and default is not callable
            setattr(type_, attr, self.default)
            return
        # attr has been assigned _reset_token and default is a callable.
        base = type_.__base__
        try:
            cache = getattr(base, attr)
        except AttributeError:
            try:
                default = self.default(type_)
            except: # If the call to default fails, the assignment is skipped
                pass
            else:
                setattr(type_, attr, default)
            return
        if cache == self.default(base):
            setattr(type_, attr, self.default(type_))
        else:
            delattr(type_, attr)

    @classmethod
    def reset(cls, type_):
        """Iteratively resets TypeAttributes on supplied type_.

        See TypeAttribute._reset for details.
        """
        for ta in type(type_).__typeattributes__.values():
            ta._reset(type_)


def typeattribute(default=None, validate=None):
    """Sets a TypeAttribute in a host archetype / prototoype.

    Params:
        default: a default value, which may be a callable that takes a
            clade type as its single parameter.
        validate: an optional validation function for attribute assignments.
    """
    return TypeAttribute(default, validate)


# Attributes for MetaCharacter
metacharacter_attrs = {'cls': nxattr.ib(init=False),
                       'name': nxattr.ib(init=False),
                       'default': nxattr.ib(default=None, init=False),
                       'validate': nxattr.ib(default=None)}


@metaregistry('__metacharacters__', reset=True)
@nxattr.s(these=metacharacter_attrs, inherit=False)
class MetaCharacter(TypeAttribute):
    """A TypeAttribute that defines a member of a clade.

    Prototypes declare metacharacters, which are used to register and
    uniquely identify types within their clade.
    Metacharacters cannot declare default values.
    """
    pass


def metacharacter(validate=None):
    """Sets a MetaCharacter in a host prototoype.

    Params:
        validate: an optional validation function for attribute assignments.
    """
    return MetaCharacter(validate)

# =============================================================================
# Typeproperties
# =============================================================================


# Attributes for TypeProperty
typeproperty_attrs = {'cls': nxattr.ib(init=False),
                      'name': nxattr.ib(init=False),
                      '__func__': nxattr.ib()}


@nxattr.s(these=typeproperty_attrs, inherit=False)
class TypeProperty(MetaDescriptor):
    """A property of a type that is accessible to type instances.

    Like with Type attributes, a regular instance property is created as well, 
    such that the type property evaluation is accessible from objects.
    The obvious differnce is that properties are computed rather than set.
    Note that this implementation is limited to read-only properties.
    """

    @property
    def _property(self):
        """Returns a property object."""
        return property(self.__func__)

    def __call__(self, mcl):
        super().__call__(mcl)
        setattr(mcl, self.name, self._property)
        inst_property = property(lambda i: getattr(type(i), self.name))
        setattr(self.cls, self.name, inst_property)


def typeproperty(func):
    """A method decorator that declares a TypeProperty."""
    return TypeProperty(func)


class CachedTypeProperty(TypeProperty):
    """A TypeProperty that is cached upon first call.

    Note that this bears strong similarities with a TypeAttribute whose
    default is a callable. However:
        * A TypeAttribute whose default is a callable can be set at
        type instantiation. That is not the case with a CachedTypeProperty.
        * Caching is effected lazily -i.e. upon first call. That is different
        from a TypeAttribute that is always set at type creation.
        * The property set upon types is a cachedproperty, which ignores
        inheritance relationships between types. That is, the cachedproperty
        is recomputed on every type irrespective of what exists in parent
        types. By contrast, TypeAttributes are passed transitively to
        subclasses.
    """

    @property
    def _property(self):
        """Returns a property object."""
        return xprops.cachedproperty(self.__func__, cache='_' + self.name)


def cachedtypeproperty(func):
    """A method decorator that declares a CachedTypeProperty."""
    return CachedTypeProperty(func)

# =============================================================================
# Typemethods
# =============================================================================


# Attributes for TypeMethod
typemethod_attrs = {'cls': nxattr.ib(init=False),
                    'name': nxattr.ib(init=False),
                    '__func__': nxattr.ib()}


@nxattr.s(these=typemethod_attrs, inherit=False)
class TypeMethod(MetaDescriptor):
    """A TypeMethod is declared in an archetype but becomes a metaclass method.

    The class is merely a transport vector to pass the method from the
    archetype to its corresponding metatype.
    """

    def __call__(self, mcl):
        super().__call__(mcl)
        setattr(mcl, self.name, self.__func__)


def typemethod(func):
    """A method decorator that declares a TypeMethod."""
    return TypeMethod(func)


class ClassTypeMethod(TypeMethod):
    """A TypeMethod that becomes a classmethod in the metaclass."""

    def __call__(self, mcl):
        super().__call__(mcl)
        setattr(mcl, self.name, classmethod(self.__func__))


def classtypemethod(func):
    """A method decorator that declares a ClassTypeMethod."""
    return ClassTypeMethod(func)


@metaregistry('__newtypemethods__')
class NewTypeMethod(TypeMethod):
    """TypeMethod that is executed at type creation.

    newtype methods are run in order of declaration as the last action in
    NxType's __new__ method. newtype methods *must return* a type,
    which is iteratively passed on to the next method until it is finally
    returned by NxType's __new__.
    """
    pass


def newtypemethod(func):
    """A method decorator that declares a NewTypeMethod."""
    return NewTypeMethod(func)


@metaregistry('__typeinitmethods__')
class TypeInitMethod(TypeMethod):
    """Typemethod that is executed at type initialization.

    typeinit methods are run in order of declaration right after a new
    subtype is registered with its archetype.
    """
    pass


def typeinitmethod(func):
    """A method decorator that declares a TypeInit method."""
    return TypeInitMethod(func)

# =============================================================================
# MetaMethod
# =============================================================================


# Attributes for MetaMethod
metamethod_attrs = {'cls': nxattr.ib(init=False),
                    'name': nxattr.ib(init=False),
                    '__func__': nxattr.ib()}


@metaregistry('__metamethods__')
@nxattr.s(these=metamethod_attrs, inherit=False)
class MetaMethod(MetaDescriptor):
    """A MetaMethod wraps a closure to generate type-specific methods.

    Methods decorated with @metamathod must return methods, which are
    automatically assigned to new subtypes of an archetype that declares
    the metamathod, unless the subtype redefines a method of the same name.
    By convention, if the decorated method is named 'f', a method
    will be created in the metaclass named '_set_f'. As the name
    implies, a call by a given type to _set_f will set f on that type,
    by running the decorated method and assign the return value.
    If the decorated method uses a trailing underscore, as in '_f', the
    metaclass method uses the same convention, i.e. '_set__f'. If the
    decorated method is a dunder method, then one underscore is skipped
    for clarity, i.e. '__f__' results in a '_set__f__' method in the
    metaclass. Usage with leading double underscore that is not dunder is
    not provided for and the results are unknown. MetaMethod provides a
    static method that computes the metaclass method name.
    Methods decorated with @metamethod should take a single argument cls.
    In that, the methods that are generated may have a different signature
    than the method declared in the archetype.
    """

    @staticmethod
    def metaname(name):
        try:
            if name[:2] == '__' and name[-2:] == '__':
                return '_set' + name
        except KeyError:
            pass
        return '_set' + '_' + 'name'

    def __call__(self, mcl):
        super().__call__(mcl)

        def setter(cls):
            """Sets a method on the supplied type.

            This is done unless cls declares an attribute of the same
            name in its dictionary.
            """
            if self.name in cls.__dict__:
                return
            method = self.__func__(cls)
            setattr(cls, self.name, method)

        setattr(mcl, self.metaname(self.name), setter)


def metamethod(func):
    """A method decorator that declares a MetaMethod."""
    return MetaMethod(func)

# =============================================================================
# MetaInstance
# =============================================================================


class MetaInstance(MetaDeclaration):
    """A facility to specify instances in class declarations.

    MetaInstance is a facilty to declare cached instances at type creation.
    The rationale for MetaInstance is to help the declaration of Marker shades
    in the data.annotations module.
    """

    def __init__(self, *args, inherit=False, use_name=False, **kwargs):
        """Supply arguments that match the declaring type's init signature.

        If inherit is True, the declaration is passed on to subclasses, which
        will create their own metainstance with the same arguments.
        If use_name is set to True, then the name of the MetaInstance is
        added as a keyword argument in the instantiation call with key 'name'.
        Alternatively, use_name can receive any string, which will be used as
        the keyword in the instantiation call.
        """
        self.args = args
        self.inherit = inherit
        self.use_name = use_name
        self.kwargs = kwargs

    def __call__(self, cls):
        """Generates an instance of the calling class."""
        if self.use_name:
            if isinstance(self.use_name, str):
                self.kwargs[self.use_name] = self.name
            else:
                self.kwargs['name'] = self.name
        return cls(*self.args, **self.kwargs)


def metainstance(*args, inherit=False, **kwargs):
    """Helper function for class declarations."""
    return MetaInstance(*args, inherit=inherit, **kwargs)

# =============================================================================
# TypeDescriptor
# =============================================================================


class TypeDescriptorRegistry(OrderedDict):
    """A container for registering TypeDescriptors at the class level.

    These are copied through class inheritance chains to provide independence.
    """

    @classmethod
    def registries(cls, type_):
        """Returns all TypeDescriptorRegistries in type_'s attributes."""
        return {k: v for k, v in type_.__dict__.items() if isinstance(v, cls)}

    def reset(self, type_, name):
        """Binds an existing instance to a type by creating a copy."""
        setattr(type_, name, self.copy())


class TypeDescriptor(MetaDeclaration):
    """A general-purpose, type-level descriptors.

    This class is intended to create descriptors on individual types. Hence
    a subtype of an archetype may declare TypeDescriptors, and unlike
    MetaDescriptors, these will not be registered in the metaclass.
    Obviously, it's a bit of a misnomer to include TypeDescriptors in the
    the metadescriptors module. This will probably call for some refactoring
    at some point.

    TypeDescriptors can optionally be registered in an OrderedDict at the
    type level by specifying a property name. This sets
    a property such that type instances can return an ordered dictionary
    of their TypeDescriptor belonging to the same registry. For instance,
    if prop is set to typedescriptors, then a class C that uses such
    descriptors has a registry __typedesriptors__ and an instance c of
    class C has a property typedescriptors that enumerates its values.
    Double underscores are automatically added.
    """
    prop = None  # An optional property name to which a class belongs.

    @classmethod
    def _registry_name(cls):
        if cls.prop is None:
            return None
        else:
            return '__' + cls.prop.strip('_') + '__'

    @classmethod
    def _property(cls):
        """Makes collecting property for types that implement descriptors."""
        if cls.prop is None:
            return None
        rgname = cls._registry_name()

        def fget(inst):
            registry = getattr(type(inst), rgname)
            return OrderedDict([(k, getattr(inst, k)) for k in registry])
        return property(fget)

    def register(self):
        """Registers a new TypeDescriptor instance."""
        if self.prop is not None:
            rgname = self._registry_name()
            try:
                registry = getattr(self.cls, rgname)
            except AttributeError:
                registry = TypeDescriptorRegistry()
                setattr(self.cls, rgname, registry)
                setattr(self.cls, self.prop, self._property())
            registry[self.name] = self

    def bind(self):
        """Binds self to its declaring class."""
        setattr(self.cls, self.name, self)
        self.register()

    # Base methods

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return None

    def __set__(self, obj, value):
        raise AttributeError("Can't set attribute.")

    def __delete__(self, obj):
        raise AttributeError("Can't delete attribute.")
