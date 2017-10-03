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
# Specifcation descriptor class
# =============================================================================


def stype_converter(stype):
    """The converter function for the stype attribute of SpecDescriptor."""
    if isinstance(stype, type):
        return stype
    try:
        return SpecType.__registry__[stype]
    except KeyError:
        if isinstance(stype, str):
            msg = "Unknown specification type registration key {0}"
            raise KeyError(msg.format(stype))
        else:
            raise TypeError


@nxattr.s
class SpecKey:
    """Container for metadata regarding a specification item."""
    name = nxattr.ib(validator=nxattr.validators.optional(
                        nxattr.validators.instance_of(str)),
                     default=None)
    stype = nxattr.ib(convert=nxattr.converters.optional(stype_converter),
                      default=None)
    required = nxattr.ib(validator=nxattr.validators.instance_of(bool),
                         default=False)
    default = nxattr.ib(default=None)
    validate = nxattr.ib(validator=nxattr.validators.optional(callable),
                         default=None)

    @xprops.weakproperty
    def descriptor(self):
        return None

    def getter(self, spec):
        """The getter method for a spec type that declares the key."""
        try:
            return spec._pull(spec._data[self.name])
        except KeyError:
            if self.default is not None:
                return self.setter(spec, self.default)

    def setter(self, spec, val):
        """The setter method for a spec type that declares the key."""
        if val is None:
            if self.required:
                msg = "Cannot set None value on required key {0}."
                raise TypeError(msg.format(self.name))
        else:
            if self.stype is not None:
                if not isinstance(val, self.stype):
                    msg = "Incorrect type {0} supplied to key {1}."
                    raise TypeError(msg.format(type(val), self.name))
            if self.validate is not None:
                try:
                    assert self.validate(val)
                except AssertionError:
                    msg = "Invalid value {0} supplied to key {1}."
                    raise ValueError(msg.format(val, self.name))
        if isinstance(val, Spec):
            spec._data.__setitem__(self.name, val._data)
        elif val is not None:
            spec._data.__setitem__(self.name, val)
        else:
            spec._data.__delitem__(self.name)
        if self.descriptor is not None:
            setattr(spec, self.descriptor.cache, val)
        return val

    def deleter(self, spec):
        """The deleter method for a spec type that declares the key."""
        if self.required:
            msg = "Cannot delete required key {0}."
            raise AttributeError(msg.format(self.name))
        try:
            del spec._data[self.name]
        except KeyError:
            pass
        if self.descriptor is not None:
            try:
                delattr(spec, self.descriptor.cache)
            except AttributeError:
                pass


class SpecDescriptor:
    """The descriptor type used in mapped specifications."""

    def __init__(self, key, name=None):
        self.key = key
        if name is not None:
            self.name = name
        else:
            self.name = key.name
        self.key.descriptor = self

    @xprops.settablecachedproperty
    def cache(self):
        return '_' + self.name

    @xprops.settablecachedproperty
    def name(self):
        return None

    @name.setter
    def name(self, val):
        if val == 'name':
            msg = "A spec descriptor cannot be named 'name'."
            raise ValueError(msg)
        setattr(self, '_name', val)
        if self.key.name is None:
            self.key.name = val

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return getattr(obj, self.cache)
        except AttributeError:
            val = self.key.getter(obj)
            setattr(obj, self.cache, val)
            return val

    def __set__(self, obj, value):
        self.key.setter(obj, value)

    def __delete__(self, obj):
        self.key.deleter(obj)


def spec(type=None, required=False, default=None, validate=None, key=None):
    """Declares a specification key and descriptor in a SpecDict.

    Params:
        type: either a string pointing to a registered SpecType, or a Python
            type. Defaults to None, which means that yaml's built-in type
            conversion applies, and eiter SpecList or SpecDict are instantiated
            recursively in case the nested contents are collections.
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
    spec_key = SpecKey(key, type, required, default, validate)
    return SpecDescriptor(spec_key)

# =============================================================================
# Specifcation metaclass
# =============================================================================


class SpecType(abc.ABCMeta):
    """The Specification metaclass."""
    __registry__ = WeakValueDictionary()

    def __new__(mcl, name, bases, namespace, etype=None, **yaml):
        cls = super().__new__(mcl, name, bases, namespace)
        mcl.__registry__[name] = cls
        return cls

    def __init__(cls, name, bases, namespace, etype=None, **yaml):
        cls._yaml = YAML(**yaml)
        if etype is not None:
            if not isinstance(etype, SpecType):
                msg = "The element type for a Spec type must be another " + \
                      "Spec type."
                raise TypeError(msg)
            cls.__etype__ = etype

    def __getitem__(cls, etype):
        """Returns a subclass with overwritten etype."""
        if not isinstance(etype, SpecType):
            msg = "The element type for a Spec type must be another " + \
                  "Spec type."
            raise TypeError(msg)
        name = etype.__name__ + cls.__name__
        bases = (cls,)
        namespace = {}
        return SpecType(name, bases, namespace, etype)


class SpecDictType(SpecType):
    """Specialized metaclass for mapped specifications."""

    def __init__(cls, name, bases, namespace, etype=None, **yaml):
        base_keys = [b.__keys__ for b in bases if isinstance(b, SpecDictType)]
        new_keys = OrderedDict()
        if 'keys' in namespace:
            for k, v in namespace['keys'].items():
                if isinstance(v, SpecKey):
                    v.name = k
                    new_keys[k] = v
        for k, v in namespace.items():
            if isinstance(v, SpecDescriptor):
                v.name = k
                key = v.key
                new_keys[key.name] = key
        cls.__keys__ = ChainMap(new_keys, *base_keys)
        super().__init__(name, bases, namespace, etype=etype)


# =============================================================================
# Base classes
# =============================================================================


# Abstract base class for List and Dict
class Spec(metaclass=SpecType):
    # Corresponding ruamel commented type
    __rtype__ = (CommentedSeq, CommentedMap)
    __etype__ = None  # Default element type for items in the Specification

    @abc.abstractmethod
    def __init__(self):
        pass

    @classmethod
    def from_rtype(cls, data):
        """Instantiates a Spec from a ruamel commented collection."""
        if not isinstance(data, cls.__rtype__):
            msg = "Method call requires a ruamel Commented collection."
            raise TypeError(msg)
        if cls is Spec:
            if isinstance(data, CommentedSeq):
                cls = SpecList
            elif isinstance(data, CommentedMap):
                cls = SpecDict
        instance = cls()
        instance._data = data
        return instance

    def _pull(self, item):
        """Converts item stored in _data to the appropriate type."""
        if self.__etype__ is not None:
            return self.__etype__.from_rtype(item)
        if isinstance(item, CommentedSeq):
            return SpecList.from_rtype(item)
        elif isinstance(item, CommentedMap):
            return SpecDict.from_rtype(item)
        else:
            return item

    def _push(self, item):
        """Converts specification item to the suitable type for ruamel."""
        if isinstance(item, Spec):
            return item._data
        else:
            return item

    @classmethod
    def load(cls, source):
        """Loads source, either a file pointer, string or pathlib.Path."""
        return cls.from_rtype(cls._yaml.load(source))

    def dump(self, sink):
        """Dumps spec to sink, either a file pointer or pathlib.Path."""
        return self._yaml.dump(self._data, sink)


class SpecList(Spec, MutableSequence):
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


class SpecDict(Spec, MutableMapping, metaclass=SpecDictType):
    """A keyed specification."""
    __rtype__ = CommentedMap
    keys = OrderedDict()  # Optional declaration of SpecKeys

    def __init__(self, mapping=(), **kwargs):
        self._data = CommentedMap()
        self.update(mapping, **kwargs)

    def __len__(self):
        stored_keys = set(self._data)
        declared_keys = set(self.__keys__)
        return len(stored_keys | declared_keys)

    def __getitem__(self, key):
        try:
            return self.__keys__[key].getter(self)
        except KeyError:
            item = self._data[key]
            return self._pull(item)
        except:
            raise KeyError

    def __setitem__(self, key, val):
        try:
            self.__keys__[key].setter(self, val)
        except KeyError:
            self._data.__setitem__(key, self._push(val))

    def __delitem__(self, key):
        try:
            self.__keys__[key].deleter(self)
        except KeyError:
            self._data.__delitem__(key)

    def __iter__(self):
        stored_keys = set(self._data)
        declared_keys = set(self.__keys__)
        return iter(stored_keys | declared_keys)
