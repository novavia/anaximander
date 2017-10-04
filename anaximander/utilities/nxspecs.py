#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides facilities for YAML-based specifications.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================


import abc
from collections import OrderedDict, ChainMap
from collections.abc import MutableSequence, MutableMapping
from ruamel.yaml.comments import CommentedSeq, CommentedMap
from weakref import WeakValueDictionary

from anaximander.utilities import nxattr, xprops


from ruamel.yaml import YAML

# =============================================================================
# Specifcation item classes
# =============================================================================


def stype_processor(stype):
    """Converts a type or string to the relevant specification type."""
    if isinstance(stype, type):
        return stype
    try:
        return SpecContainerType.__registry__[stype]
    except KeyError:
        if isinstance(stype, str):
            msg = "Unknown specification container type registration key {0}"
            raise KeyError(msg.format(stype))
        else:
            raise TypeError


def _validate_validator(inst, attr, value):
    if not(value is None or callable(value)):
        msg = "Improper validation function {0}"
        raise ValueError(msg.format(value))


@nxattr.s
class Spec:
    """Container for metadata regarding a specification item."""
    stype = nxattr.ib(default=None)
    default = nxattr.ib(default=None)
    validate = nxattr.ib(validator=_validate_validator, default=None)

    @xprops.cachedproperty
    def container(self):
        return isinstance(self.spec_type, SpecContainerType)

    @xprops.cachedproperty
    def spec_type(self):
        """The actual spec type, inferred from stype at runtime."""
        stype = self.stype
        if stype is None:
            return None
        if isinstance(stype, type):
            return stype
        try:
            return SpecContainerType.__registry__[stype]
        except KeyError:
            if isinstance(stype, str):
                msg = "Unknown specification container type name {0}"
                raise KeyError(msg.format(stype))
            else:
                raise TypeError

    def _validate(self, val):
        if self.spec_type is not None:
            if not isinstance(val, self.spec_type):
                msg = "Incorrect type {0} supplied to {1}."
                raise TypeError(msg.format(type(val), self))
        if self.validate is not None:
            try:
                assert self.validate(val)
            except AssertionError:
                msg = "Invalid value {0} supplied to {1}."
                raise ValueError(msg.format(val, self))

    def load(self, val):
        """Loads data from a yaml collection."""
        if val is None:
            if self.default is not None:
                return self.default
        if self.container:
            val = self.spec_type.from_rtype(val)
        self._validate(val)
        return val

    def dump(self, val):
        """Passes data to a yaml collection."""
        if val is None:
            return
        self._validate(val)
        if isinstance(val, SpecContainer):
            return val._data
        return val


class EnumerationSpec(Spec):

    def __init__(self, *enumeration, default=None):
        super().__init__(validate=lambda x: x in enumeration, default=default)


@nxattr.s
class KeyedSpec:
    """Container for metadata regarding a keyed specification item."""
    spec = nxattr.ib(validator=nxattr.validators.instance_of(Spec))
    key = nxattr.ib(validator=nxattr.validators.optional(
                        nxattr.validators.instance_of(str)),
                    default=None)
    required = nxattr.ib(validator=nxattr.validators.instance_of(bool),
                         default=False)

    def getter(self, container):
        """The getter method for a container that declares the key."""
        try:
            val = container._data[self.key]
        except KeyError:
            if self.spec.default is not None:
                return self.setter(container, self.spec.default)
        else:
            return self.spec.load(val)

    def setter(self, container, val):
        """The setter method for a container that declares the key."""
        if val is None:
            if self.required:
                msg = "Cannot set None value on required key {0}."
                raise TypeError(msg.format(self.key))
        else:
            val = self.spec.dump(val)
        if val is not None:
            container._data.__setitem__(self.key, val)
        else:
            container._data.__delitem__(self.key)
        if isinstance(self, SpecDescriptor):
            setattr(container, self.cache, val)

    def deleter(self, container):
        """The deleter method for a container that declares the key."""
        if self.required:
            msg = "Cannot delete required key {0}."
            raise AttributeError(msg.format(self.key))
        try:
            del container._data[self.key]
        except KeyError:
            pass
        if isinstance(self, SpecDescriptor):
            try:
                del self.cache
            except AttributeError:
                pass


class SpecDescriptor(KeyedSpec):
    """The descriptor type used in mapped specifications."""

    @xprops.settablecachedproperty
    def cache(self):
        return '_' + self.attr

    @xprops.settablecachedproperty
    def attr(self):
        """Attribute name."""
        return self.key

    @attr.setter
    def name(self, val):
        setattr(self, '_attr', val)
        if self.key is None:
            self.key = val

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return getattr(obj, self.cache)
        except AttributeError:
            val = self.getter(obj)
            setattr(obj, self.cache, val)
            return val

    def __set__(self, obj, value):
        self.setter(obj, value)

    def __delete__(self, obj):
        self.deleter(obj)


def spec(type=None, required=False, default=None, validate=None, key=None):
    """Declares a specification key and descriptor in a SpecDict.

    Params:
        type: one of the following object types:
            * None is valid
            * A type, including in particular a SpecContainerType
            * A Spec instance
            * A string that registers a SpecContainerType
            If None, any type will be admitted -though this could be overriden
            by the validate argument. Note that yaml's built-in type
            conversion applies, such that collection will be automatically
            and recursively turned into either SpecList or SpecDict.
            If a type, then type validation is applied, irrespective of
            what is passed to validate.
            In the case of SpecContainerType, a string can be supplied, which
            be looked up in the SpecContainerType registry.
            If a Spec instance, then the descriptor will expect values to
            conform to the Spec. Note that this is basically a different
            way to pass the specification arguments that are included in the
            function call, since in every other case a Spec instance is
            created. If other arguments normally used to generate a Spec
            are passed an error is raised.
        required: boolean. If True, specification maps must always provide
            a value for the corresponding key.
        default: a default value that is used if no value is provided for
            the corresponding key.
        validate: either None or a callable that must return True on a
            valid value.
        key: optionally, a string name to use as the specification storage
            key. This could be different from the name supplied to the
            descriptor, especially if it is supposed to contain spaces and/or
            capital letters. For instance the key "Device Type" could be
            declared with a descriptor device_type, and the declaration would
            need to supply the string "Device Type" to the key argument. If
            unspecified, the key is identical to the descriptor name.

    Returns:
        a SpecDescriptor object, which is interpreted by the metaclass -in
        particular, the metaclass provides name assignment.
    """
    if isinstance(type, Spec):
        spec_instance = type
        if any([default, validate]):
            raise TypeError()
    else:
        spec_instance = Spec(type, default, validate)
    return SpecDescriptor(spec_instance, key, required)

# =============================================================================
# Specification container classes
# =============================================================================


class SpecContainerType(abc.ABCMeta):
    """The Specification container metaclass.

    The metaclass takes two original inputs:
        * spec: Either a Spec instance, or a container type or string
            argument that can readily be supplied to Spec. This sets the
            default container item type, and can be left to None.
        * **yaml: arguments to pass to the creation of a YAML object used
            for serialization / deserialization.
    """
    __registry__ = WeakValueDictionary()

    def __new__(mcl, name, bases, namespace, spec=None, **yaml):
        cls = super().__new__(mcl, name, bases, namespace)
        mcl.__registry__[name] = cls
        return cls

    def __init__(cls, name, bases, namespace, spec=None, **yaml):
        cls._yaml = YAML(**yaml)
        if spec is not None:
            if not isinstance(spec, Spec):
                spec = Spec(spec)
            cls.__spec__ = spec

    def __getitem__(cls, spec):
        """Returns a subclass with overwritten spec."""
        return SpecContainerType(cls.__name__, (cls,), {}, spec=spec)


class SpecDictType(SpecContainerType):
    """Specialized metaclass for mapped specifications."""

    def __init__(cls, name, bases, namespace, spec=None, **yaml):
        keyspecs = [b.__keyspecs__ for b in bases
                    if isinstance(b, SpecDictType)]
        new_keyspecs = OrderedDict()
        if 'keyspecs' in namespace:
            for k, v in namespace['keyspecs'].items():
                if isinstance(v, KeyedSpec):
                    v.key = k
                    new_keyspecs[k] = v
        for k, v in namespace.items():
            if isinstance(v, SpecDescriptor):
                v.attr = k
                new_keyspecs[v.key] = v
        cls.__keyspecs__ = ChainMap(new_keyspecs, *keyspecs)
        super().__init__(name, bases, namespace, spec=spec)


# Abstract base class for List and Dict
class SpecContainer(metaclass=SpecContainerType):
    # Corresponding ruamel commented type(s)
    __rtype__ = (CommentedSeq, CommentedMap)
    __spec__ = None  # Placeholder for specifying default element type

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def spec(self):
        return type(self).__spec__

    @classmethod
    def from_rtype(cls, data):
        """Instantiates a Spec from a ruamel commented collection."""
        if not isinstance(data, cls.__rtype__):
            msg = "Method call requires a ruamel Commented collection."
            raise TypeError(msg)
        if cls is SpecContainer:
            if isinstance(data, CommentedSeq):
                cls = SpecList
            elif isinstance(data, CommentedMap):
                cls = SpecDict
        instance = cls()
        instance._data = data
        return instance

    def _pull(self, val):
        """Converts yaml content to suitable type."""
        if self.spec is not None:
            return self.spec.load(val)
        if isinstance(val, CommentedSeq):
            return SpecList.from_rtype(val)
        elif isinstance(val, CommentedMap):
            return SpecDict.from_rtype(val)
        else:
            return val

    def _push(self, val):
        """Converts specification item to the suitable type for ruamel."""
        if self.spec is not None:
            return self.spec.dump(val)
        if isinstance(val, SpecContainer):
            return val._data
        else:
            return val

    @classmethod
    def load(cls, source):
        """Loads source, either a file pointer, string or pathlib.Path."""
        return cls.from_rtype(cls._yaml.load(source))

    def dump(self, sink):
        """Dumps spec to sink, either a file pointer or pathlib.Path."""
        return self._yaml.dump(self._data, sink)


class SpecList(SpecContainer, MutableSequence):
    """An enumerated specification."""
    __rtype__ = CommentedSeq

    def __init__(self, *args):
        self._data = CommentedSeq()
        self.extend(args)

    def __len__(self):
        return self._data.__len__()

    def __getitem__(self, ix):
        return self._pull(self._data[ix])

    def __setitem__(self, ix, val):
        self._data.__setitem__(ix, self._push(val))

    def __delitem__(self, ix):
        self._data.__delitem__(ix)

    def insert(self, ix, val):
        self._data.insert(ix, self._push(val))


class SpecDict(SpecContainer, MutableMapping, metaclass=SpecDictType):
    """A keyed specification."""
    __rtype__ = CommentedMap
    specs = OrderedDict()  # Optional declaration of SpecKeys

    def __init__(self, mapping=(), **kwargs):
        self._data = CommentedMap()
        self.update(mapping, **kwargs)

    def __len__(self):
        stored_keys = set(self._data)
        declared_keys = set(self.__keyspecs__)
        return len(stored_keys | declared_keys)

    def __getitem__(self, key):
        try:
            return self.__keyspecs__[key].getter(self)
        except KeyError:
            item = self._data[key]
            return self._pull(item)
        except:
            raise KeyError

    def __setitem__(self, key, val):
        try:
            self.__keyspecs__[key].setter(self, val)
        except KeyError:
            self._data.__setitem__(key, self._push(val))

    def __delitem__(self, key):
        try:
            self.__keyspecs__[key].deleter(self)
        except KeyError:
            self._data.__delitem__(key)

    def __iter__(self):
        stored_keys = set(self._data)
        declared_keys = set(self.__keyspecs__)
        return iter(stored_keys | declared_keys)
