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

from anaximander.utilities import xprops

# =============================================================================
# Base metadescriptors
# =============================================================================


class MetaDescriptorError(Exception):
    """Customized error class for MetaDescriptor errors."""
    pass


class MetaDescriptor(abc.ABC):
    """Base class for metadescriptors.

    Metadescriptors are intended to be inserted in type declarations
    to modify behavior for derived types. Their scope is types rather than
    objects, and thus they are declarative devices that get processed to
    become descriptors in a metaclass, hence the name metadescriptor.
    The NxType base metaclass systematically collects metadescriptors found
    in type declarations, strips them from the type's namespace, and park
    them into a __metadescriptors__ dictionary. For regular types this
    accomplishes nothing, but if a type is decorated with @archetype or
    @prototype, then the __metadescriptors__ dictionary is interpreted in
    order to create a new metaclass that implements the behaviors programmed
    in the metadescriptors.
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

    @abc.abstractmethod
    def __call__(self, mcl):
        """Adds targeted behavior to the supplied metaclass."""
        if self.cls is None or self.name is None:
            msg = "Cannot call unbound metadescriptor instance."
            raise TypeError(msg)


class ValidationError(MetaDescriptorError, ValueError):
    """Raised if wrong value passed to a TypeAttribute."""
    pass


class TypeAttribute(MetaDescriptor):
    """A metadescriptor that sets a type keyword argument."""

    def __init__(self, default=None, validate=None):
        """Instantiates a TypeAttribute.

        params:
            validate (func): a validation function that will run on
                values supplied to new types for the type attribute.
                Should simply return True upon success.
            default: a default value that is passed to new types in
                case no value is supplied.
        """
        self.default = default
        self.validate = validate

    def __call__(self, mcl):
        mcl.__typeattributes__[self.name] = self
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


class MetaCharacter(TypeAttribute):
    """A TypeAttribute that defines a member of a clade.

    Archetypes and prototypes declare metacharacters, which are used to
    register and uniquely identify types within their clade.
    """

    def __call__(self, mcl):
        super().__call__(mcl)
        mcl.__metacharacters__[self.name] = self
