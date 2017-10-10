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
import datetime as dt
from weakref import WeakValueDictionary

import pandas as pd
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq, CommentedMap

from anaximander.utilities import xprops

# =============================================================================
# Specifcation item classes
# =============================================================================


class Spec(abc.ABC):
    """Container for metadata regarding a specification item.

    Params:
        stype: one of the following object types:
            * None is valid
            * A type, including in particular a SpecContainerType
            * A string that registers a SpecContainerType
            If None, any type will be admitted -though this could be overriden
            by the validate argument. Note that yaml's built-in type
            conversion applies, such that collection will be automatically
            and recursively turned into either SpecList or SpecDict.
            If a type, then type validation is applied, irrespective of
            what is passed to validate.
            In the case of SpecContainerType, a string can be supplied, which
            be looked up in the SpecContainerType registry.
        key: optionally, a string name to use as the specification storage
            key. This is only used in SpecDict (SpecList instances don't
            implement keys). If a spec is declared as a class descriptor,
            the descriptor's attribute name can be different from the key.
            This is especially useful to specify keys containing spaces and/or
            capital letters. For instance the key "Device Type" could be
            declared with a descriptor device_type, and the declaration would
            need to supply the string "Device Type" to the key argument. If
            unspecified, the key is identical to the descriptor name.
        default: a default value that is used if no value is provided for
            the corresponding key.
        validator: either None or a callable that must return True on a
            valid value.
        required: boolean. If True, specification maps must always provide
            a value for the corresponding key.
    """

    def __init__(self, stype=None, key=None, default=None, validator=None,
                 required=False):
        self.stype = stype
        if key is not None:
            self.key = key
        self.default = default
        self.validator = validator
        self.required = required

    @xprops.singlesetproperty
    def stype(self):
        return None

    @xprops.singlesetproperty
    def key(self):
        return None

    @xprops.singlesetproperty
    def default(self):
        return None

    @xprops.singlesetproperty
    def validator(self):
        return None

    @xprops.singlesetproperty
    def required(self):
        return None

    @xprops.singlesetproperty
    def attr(self):
        """Attribute name."""
        return self.key

    @attr.setter
    def attr(self, val):
        setattr(self, '_attr', val)
        if not hasattr(self, '_key'):
            self.key = val

    @property
    def cache(self):
        return '_' + self.attr

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

    @property
    def container(self):
        """True if the spec designates a nested container."""
        return isinstance(self.spec_type, SpecContainerType)

    @xprops.settablecachedproperty
    def descriptor(self):
        """True if the spec was declared as a class descriptor."""
        return False

    def getter(self, container):
        """The getter method for a SpecDict that declares the spec."""
        if self.key is None or not isinstance(container, SpecDict):
            msg = "Call is only permitted with a keyed specification."
            raise TypeError(msg)
        try:
            val = container._data[self.key]
        except KeyError:
            if self.default is not None:
                return self.setter(container, self.default)
        else:
            return self.load(val)

    def setter(self, container, val):
        """The setter method for a SpecDict that declares the spec."""
        if self.key is None or not isinstance(container, SpecDict):
            msg = "Call is only permitted with a keyed specification."
            raise TypeError(msg)
        if val is None:
            if self.required:
                msg = "Cannot set None value on required key {0}."
                raise TypeError(msg.format(self.key))
        else:
            val = self.dump(val)
        if val is not None:
            container._data.__setitem__(self.key, val)
        else:
            container._data.__delitem__(self.key)
        if self.descriptor:
            setattr(container, self.cache, val)

    def deleter(self, container):
        """The deleter method for a SpecDict that declares the spec."""
        if self.key is None or not isinstance(container, SpecDict):
            msg = "Call is only permitted with a keyed specification."
            raise TypeError(msg)
        if self.required:
            msg = "Cannot delete required key {0}."
            raise AttributeError(msg.format(self.key))
        try:
            del container._data[self.key]
        except KeyError:
            pass
        if self.descriptor:
            try:
                del self.cache
            except AttributeError:
                pass

    def __validator__(self, val):
        """An optional validator method for subclasses."""
        return True

    def __loader__(self, val):
        """An optional loader method for subclasses.

        The method is applied on values that have already been deserialized
        from a YAML specification, and offers the opportunity for additional
        transformation.
        Note that this does not operate on container specifications.
        """
        return val

    def __dumper__(self, val):
        """An optional dumper method for subclasses.

        Inputs are passed from specification container instances and the
        return value is supplied to the YAML serializer.
        Note that this does not operate on container specifications.
        """
        return val

    def validate(self, val):
        if self.spec_type is not None:
            if not isinstance(val, self.spec_type):
                msg = "Incorrect type {0} supplied to {1}."
                raise TypeError(msg.format(type(val), self))
        try:
            assert self.__validator__(val)
        except AssertionError:
            msg = "Invalid value {0} supplied to {1}."
            raise ValueError(msg.format(val, self))
        if self.validator is not None:
            try:
                assert self.validator(val)
            except AssertionError:
                msg = "Invalid value {0} supplied to {1}."
                raise ValueError(msg.format(val, self))

    def load(self, val):
        """Loads data from a yaml collection."""
        if self.container:
            rval = self.spec_type.from_rtype(val)
        elif val is None and self.default is not None:
            rval = self.default
        else:
            rval = self.__loader__(val)
        self.validate(rval)
        return rval

    def dump(self, val):
        """Passes data to a yaml collection."""
        if val is None:
            return
        self.validate(val)
        if isinstance(val, SpecContainer):
            return val._data
        return self.__dumper__(val)

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

    def __repr__(self):
        prefix = type(self).__name__
        try:
            stype = self.spec_type.__name__
        except AttributeError:
            stype = None
        string = '{p}(type={t}, key={k}, default={d})'
        return string.format(p=prefix, t=stype, k=self.key, d=self.default)


class ContainerSpec(Spec):
    """Base class for List and Dict specifications.

    The signature is modified slightly such that the first optional argument
    specifies the default expected type of the container's items. Admissible
    values for ispec are the same as those passed to SpecContainerType, i.e.
    either a Spec instance, a SpecContainer type, or a SpecContainery type
    name.
    """

    def __init__(self, ispec=None, key=None, default=None, validator=None,
                 required=False):
        self.ispec = ispec
        if key is not None:
            self.key = key
        self.default = default
        self.validator = validator
        self.required = required

    @xprops.singlesetproperty
    def ispec(self):
        return None

    @abc.abstractproperty
    def stype(self):
        return None


class List(ContainerSpec):

    @property
    def stype(self):
        return SpecList[self.ispec]


class Dict(ContainerSpec):

    @property
    def stype(self):
        return SpecDict[self.ispec]


class TypedSpec(Spec):
    """A Spec whose type is defined at the class level."""

    def __init__(self, key=None, default=None, validator=None, required=False):
        if key is not None:
            self.key = key
        self.default = default
        self.validator = validator
        self.required = required

    @abc.abstractproperty
    def stype(self):
        return None


class Str(TypedSpec):

    @property
    def stype(self):
        return str


class Int(TypedSpec):

    @property
    def stype(self):
        return int


class Float(TypedSpec):

    @property
    def stype(self):
        return float


class Date(TypedSpec):

    @property
    def stype(self):
        return dt.date


class DateTime(TypedSpec):

    @property
    def stype(self):
        return dt.datetime


class Timestamp(TypedSpec):

    @property
    def stype(self):
        return pd.Timestamp

    def __loader__(self, val):
        return pd.Timestamp(val)

    def __dumper__(self, val):
        return val.to_pydatetime()


class Selection(Str):
    enumeration = []

    def __init__(self, *enumeration, key=None, default=None, validator=None,
                 required=False):
        if enumeration:
            self.enumeration = enumeration
        super().__init__(key, default, validator, required)

    def __validator__(self, val):
        return val in self.enumeration

# =============================================================================
# Specification container classes
# =============================================================================


class SpecContainerType(abc.ABCMeta):
    """The Specification container metaclass.

    The metaclass takes two original inputs:
        * ispec: Either a Spec instance, or a container type or the name
            of a container type which is looked up. This sets the
            default container item type, and can be left to None.
        * **yaml: arguments to pass to the creation of a YAML object used
            for serialization / deserialization.
    """
    __registry__ = WeakValueDictionary()

    def __new__(mcl, name, bases, namespace, ispec=None, **yaml):
        cls = super().__new__(mcl, name, bases, namespace)
        mcl.__registry__[name] = cls
        return cls

    def __init__(cls, name, bases, namespace, ispec=None, **yaml):
        cls._yaml = YAML(**yaml)
        if ispec is not None:
            if not isinstance(ispec, Spec):
                ispec = Spec(ispec)
            cls.__ispec__ = ispec

    def __getitem__(cls, ispec):
        """Returns a subclass with overwritten spec."""
        return SpecContainerType(cls.__name__, (cls,), {}, ispec=ispec)


class SpecDictType(SpecContainerType):
    """Specialized metaclass for mapped specifications."""

    def __init__(cls, name, bases, namespace, ispec=None, **yaml):
        keyspecs = [b.__keyspecs__ for b in bases
                    if isinstance(b, SpecDictType)]
        new_keyspecs = OrderedDict()
        if 'keyspecs' in namespace:
            for k, v in namespace['keyspecs'].items():
                if isinstance(v, Spec):
                    v.key = k
                    new_keyspecs[k] = v
        for k, v in namespace.items():
            if isinstance(v, Spec):
                v.attr = k
                v.descriptor = True
                new_keyspecs[v.key] = v
        cls.__keyspecs__ = ChainMap(new_keyspecs, *keyspecs)
        super().__init__(name, bases, namespace, ispec=ispec)


# Abstract base class for List and Dict
class SpecContainer(metaclass=SpecContainerType):
    # Corresponding ruamel commented type(s)
    __rtype__ = (CommentedSeq, CommentedMap)
    __ispec__ = None  # Placeholder for specifying default element type

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def ispec(self):
        return type(self).__ispec__

    @xprops.singlesetproperty
    def initialized(self):
        """Lock used to start validation."""
        return False

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
        if self.ispec is not None:
            return self.ispec.load(val)
        if isinstance(val, CommentedSeq):
            return SpecList.from_rtype(val)
        elif isinstance(val, CommentedMap):
            return SpecDict.from_rtype(val)
        else:
            return val

    def _push(self, val):
        """Converts specification item to the suitable type for ruamel."""
        if self.ispec is not None:
            return self.ispec.dump(val)
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

    def __validator__(self):
        """Placeholder for a container-level validation function."""
        return True

    def validate(self):
        """Validates that self conforms to __validator__."""
        if not self.initialized:
            return
        try:
            assert self.__validator__()
        except AssertionError:
            msg = "Invalid container {0}"
            raise ValueError(msg.format(self))


class SpecList(SpecContainer, MutableSequence):
    """An enumerated specification."""
    __rtype__ = CommentedSeq

    def __init__(self, *args):
        self._data = CommentedSeq()
        self.extend(args)
        self.initialized = True

    def __len__(self):
        return self._data.__len__()

    def __getitem__(self, ix):
        return self._pull(self._data[ix])

    def __setitem__(self, ix, val):
        self._data.__setitem__(ix, self._push(val))
        self.validate()

    def __delitem__(self, ix):
        self._data.__delitem__(ix)
        self.validate()

    def insert(self, ix, val):
        self._data.insert(ix, self._push(val))
        self.validate()


class SpecDict(SpecContainer, MutableMapping, metaclass=SpecDictType):
    """A keyed specification."""
    __rtype__ = CommentedMap
    specs = OrderedDict()  # Optional declaration of SpecKeys

    def __init__(self, mapping=(), **kwargs):
        self._data = CommentedMap()
        self.update(mapping, **kwargs)
        self.initialized = True

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
        self.validate()

    def __delitem__(self, key):
        try:
            self.__keyspecs__[key].deleter(self)
        except KeyError:
            self._data.__delitem__(key)
        self.validate()

    def __iter__(self):
        stored_keys = set(self._data)
        declared_keys = set(self.__keyspecs__)
        return iter(stored_keys | declared_keys)
