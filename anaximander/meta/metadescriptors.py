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

from anaximander.utilities import xprops, nxattr


__all__ = ['TypeAttribute', 'MetaCharacter', 'metamethod', 'typeinitmethod']

# =============================================================================
# Utilities
# =============================================================================


# Container for metadescriptor registries found in anaximander metaclasses
# Metaregistries maps MetaDescriptor types to a MetaRegistryFactory instance.
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


# Attributes for MetaDescriptor
metadescriptor_attrs = {'cls': nxattr.ib(init=False),
                        'name': nxattr.ib(init=False)}


@metaregistry('__metadescriptors__')
@nxattr.s(these=metadescriptor_attrs, init=False)
class MetaDescriptor(abc.ABC):
    """Base class for metadescriptors.

    Metadescriptors are intended to be inserted in type declarations
    to modify behavior for derived types. Their scope is types rather than
    objects, and thus they are declarative devices that get processed to
    become descriptors in a metaclass, hence the name metadescriptor.
    The NxType base metaclass systematically collects metadescriptors found
    in type declarations, strips them from the type's namespace, and park
    them into a __metadeclarations__ dictionary. For regular types this
    accomplishes nothing, but if a type is decorated with @archetype then
    the __metadeclarations__ dictionary is interpreted in order to create a
    new metaclass that implements the behaviors programmed in the
    metadescriptors.
    Metadescriptor behavior is bound to a metaclass in two stages. In the
    first stage, NxType passes the declaring type to each metadescriptor.
    In the second stage, the __call__ method of the metadescriptor instance
    is called and supplied a metaclass whose dictionary gets modified as
    a result.
    """

    @xprops.singlesetproperty
    def cls(self):
        """The declaring class, set by NxType."""
        return None

    @xprops.singlesetproperty
    def name(self):
        """The declared name, set by NxType."""
        return None

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
    """

    def __call__(self, mcl):
        super().__call__(mcl)
        if callable(self.default):
            type_property = xprops.cachedproperty(lambda c: self.default(c))
        else:
            type_property = xprops.cachedproperty(lambda c: self.default)
        type_property.cache = '_' + self.name
        setattr(mcl, self.name, type_property)
        inst_property = property(lambda i: getattr(type(i), self.name))
        setattr(self.cls, self.name, inst_property)

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
                pass
            else:
                v.assign(namespace, value)


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
    """
    pass

# =============================================================================
# Metamethods
# =============================================================================


# Attributes for MetaMethod
metamethod_attrs = {'cls': nxattr.ib(init=False),
                    'name': nxattr.ib(init=False),
                    '__func__': nxattr.ib()}


@nxattr.s(these=metamethod_attrs, inherit=False)
class MetaMethod(MetaDescriptor):
    """A metamethod is declared in an archetype but becomes a metaclass method.

    The class is merely a transport vector to pass the method from the
    archetype to its corresponding metatype.
    """

    def __call__(self, mcl):
        super().__call__(mcl)
        setattr(mcl, self.name, self.__func__)


def metamethod(func):
    """A method decorator that declares a MetaMethod."""
    return MetaMethod(func)


@metaregistry('__newtypemethods__')
class NewTypeMethod(MetaMethod):
    """Metamethod that is executed at type creation.

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
class TypeInitMethod(MetaMethod):
    """Metamethod that is executed at type initialization.

    typeinit methods are run in order of declaration right after a new
    subtype is registered with its archetype.
    """
    pass


def typeinitmethod(func):
    """A method decorator that declares a TypeInit method."""
    return TypeInitMethod(func)
