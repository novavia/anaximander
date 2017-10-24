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
from collections.abc import Mapping, MutableSequence, MutableMapping
import datetime as dt
import io
import json
import os
from pathlib import Path
import shutil
from threading import Thread
import time
from weakref import WeakValueDictionary

from google.cloud.storage.client import Client
from google.cloud.exceptions import NotFound
import pandas as pd
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq, CommentedMap

from anaximander.utilities import xprops
from anaximander.utilities import datastore as dts

# =============================================================================
# Specifcation item classes
# =============================================================================


class SpecEncoder(json.JSONEncoder):
    """Custom JSON encoder for specifications."""

    def default(self, obj):
        if isinstance(obj, SpecDict):
            specs = obj.__keyspecs__
            ispec = obj.ispec
            rdict = {}
            for key, val in obj.items():
                try:
                    spec = specs[key]
                except KeyError:
                    k = key
                    if ispec:
                        v = ispec.__json__(val)
                    else:
                        v = val
                else:
                    k = spec.compact
                    v = spec.__json__(val)
                rdict[k] = v
            return rdict
        elif isinstance(obj, SpecList):
            ispec = obj.ispec
            if ispec:
                return [ispec.__json__(v) for v in obj]
            else:
                return list(obj)
        return super().default(self, obj)


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
        nullable: if True, the spec can be explicitly set to None, which
            is serialized as 'null'. Otherwise, setting to None may
            raise an error if the stype is set or if validator doesn't
            allow None values.
        compact: either None or a string. This is the key to use for compact
            representation, which is the default for JSON export.
    """

    def __init__(self, stype=None, key=None, default=None, validator=None,
                 required=False, nullable=False, compact=None):
        self.stype = stype
        if key is not None:
            self.key = key
        self.default = default
        self.validator = validator
        self.required = required
        self.nullable = nullable
        if compact is not None:
            self.compact = compact

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
    def compact(self):
        return self.key

    @xprops.singlesetproperty
    def attr(self):
        """Attribute name."""
        return self.key

    @attr.setter
    def attr(self, val):
        setattr(self, '_attr', val)
        if not hasattr(self, '_key'):
            self.key = val

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
            if not self.required:
                return self.default
            else:
                raise
        else:
            return self.load(val)

    def setter(self, container, val):
        """The setter method for a SpecDict that declares the spec."""
        if self.key is None or not isinstance(container, SpecDict):
            msg = "Call is only permitted with a keyed specification."
            raise TypeError(msg)
        val = self.dump(self.__setter__(val))
        container._data.__setitem__(self.key, val)

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

    def __setter__(self, val):
        """An optional setter method applied when setting a value."""
        if self.container:
            return self.spec_type(val)
        return val

    def __json__(self, val):
        """An optional json encoder."""
        return val

    def validate(self, val):
        if self.nullable and val is None:
            return True
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
        else:
            rval = self.__loader__(val)
        self.validate(rval)
        return rval

    def dump(self, val):
        """Passes data to a yaml collection."""
        self.validate(val)
        if isinstance(val, SpecContainer):
            return val._data
        return self.__dumper__(val)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.getter(obj)

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
                 required=False, nullable=False, compact=None):
        self.ispec = ispec
        if key is not None:
            self.key = key
        self.default = default
        self.validator = validator
        self.required = required
        self.nullable = nullable
        if compact is not None:
            self.compact = compact

    @xprops.singlesetproperty
    def ispec(self):
        return None

    @abc.abstractproperty
    def stype(self):
        return None


class List(ContainerSpec):

    @xprops.cachedproperty
    def stype(self):
        return SpecList.sub(self.ispec)


class Dict(ContainerSpec):

    @xprops.cachedproperty
    def stype(self):
        return SpecDict.sub(self.ispec)


class TypedSpec(Spec):
    """A Spec whose type is defined at the class level."""

    def __init__(self, key=None, default=None, validator=None, required=False,
                 nullable=False, compact=None):
        if key is not None:
            self.key = key
        self.default = default
        self.validator = validator
        self.required = required
        self.nullable = nullable
        if compact is not None:
            self.compact = compact

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

    def __setter__(self, val):
        if val is not None:
            return int(val)


class Float(TypedSpec):

    @property
    def stype(self):
        return float

    def __setter__(self, val):
        if val is not None:
            return float(val)


class Date(TypedSpec):

    @property
    def stype(self):
        return dt.date

    def __json__(self, val):
        return str(val)


class DateTime(TypedSpec):

    @property
    def stype(self):
        return dt.datetime

    def __json__(self, val):
        return str(val)


class Timestamp(TypedSpec):

    @property
    def stype(self):
        return pd.Timestamp

    def __loader__(self, val):
        return pd.Timestamp(val)

    def __dumper__(self, val):
        return val.to_pydatetime()

    def __setter__(self, val):
        return pd.Timestamp(val)

    def __json__(self, val):
        return str(val)


class Selection(Str):
    """Specifies a categorical variable.

    The possible categories are specified in the enumeration variable,
    either in a subclass, or optionally at instantiation.
    The enumeration can be either a sequence or a mapping. A sequence
    defines the possible values of the specification. In the case of a
    mapping, these values are the keys, whereas the mapping's values
    define the shorthand versions that is used when exporting to JSON.
    """
    enumeration = []

    def __init__(self, key=None, default=None, validator=None,
                 required=False, nullable=False, compact=None,
                 enumeration=None):
        if enumeration:
            self.enumeration = enumeration
        super().__init__(key, default, validator, required, nullable, compact)

    @property
    def mapping(self):
        return isinstance(self.enumeration, Mapping)

    def __validator__(self, val):
        return val in self.enumeration

    def __json__(self, val):
        if self.mapping:
            return self.enumeration[val]
        return val

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
    __registry__ = WeakValueDictionary()  # General type registry
    __path_registry__ = WeakValueDictionary()  # Registry for storage types

    def __new__(mcl, name, bases, namespace, ispec=None, **yaml):
        cls = super().__new__(mcl, name, bases, namespace)
        mcl.__registry__[name] = cls
        try:
            mcl.__path_registry__[namespace['__path__']] = cls
        except KeyError:
            pass
        return cls

    def __init__(cls, name, bases, namespace, ispec=None, **yaml):
        cls._yaml = YAML(**yaml)
        if ispec is not None:
            if issubclass(ispec, Spec):
                ispec = ispec()
            elif not isinstance(ispec, Spec):
                ispec = Spec(ispec)
            cls.__ispec__ = ispec
        cls.__cache__ = WeakValueDictionary()

    def __getitem__(cls, identifier):
        """Cache retrieval mechanism."""
        return cls.__cache__.__getitem__(identifier)

    def sub(cls, ispec):
        """Returns a subclass with the supplied ispec."""
        return type(cls)(cls.__name__, (cls,), {}, ispec)


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
        # Sets the identifier property as applicable
        if isinstance(cls.__identifier__, str):
            attr = cls.__identifier__
            cls.identifier = property(lambda i: getattr(i, attr))
        elif callable(cls.__identifier__):
            func = cls.__identifier__
            cls.identifier = property(func)


# Abstract base class for List and Dict
class SpecContainer(metaclass=SpecContainerType):
    """Base class for specification containers.

    SpecContainer gets subclassed into SpecList and SpecDict. The former
    is a sequence of items, which may be homogeneous or heterogeneous in
    terms of item types. The latter is expected to be much more common,
    and in particular, it allows for the use of descriptors in subclasses to
    outline content.
    Both types accept an optional ispec parameter from their metaclass, which
    indicates an expected default type for random items.
    The __path__ variable is intended to be assigned a string by subclasses,
    possibly containing slashes, so as to provide a storage path from
    a given directory root.
    The storage model is as follows: root/<owner>/<path>/<identifier>.yaml,
    where owner is an object that must print to a string which can be used
    in a file path, path is provided by the container type, and identity
    is an optional string argument. Owner is stored as a weak reference.
    If any of the three attributes owner, path or identifier evaluates to
    None, an attempt to store the specification will fail.

    Params:
        owner: optional owner. Must be an object that accepts weak references,
            and have a __str__ method that evaluates to a string compatible
            with a hierarchical file storage structure.
        identifier: an optional string that uniquely identifies the container
            within its type.
        _delay: internal parameter to enable 'from_rtype' instantiation.
    """
    # Corresponding ruamel commented type(s)
    __rtype__ = (CommentedSeq, CommentedMap)
    __ispec__ = None  # Placeholder for specifying default element type
    __path__ = None  # An optional storage path into a specification store

    @abc.abstractmethod
    def __init__(self, owner=None, identifier=None, _delay=False):
        self.owner = owner
        if identifier is not None:
            self.identifier = identifier
        if not _delay:
            self._post_init()

    def _post_init(self):
        self.initialized = True
        self.validate()

    @property
    def ispec(self):
        return type(self).__ispec__

    @xprops.singlesetproperty
    def initialized(self):
        """Lock used to start validation."""
        return False

    @xprops.weakproperty
    def owner(self):
        """Optional owner object for a specification container."""
        return None

    @xprops.singlesetproperty
    def identifier(self):
        """Optional instance identifier, unique to a container type."""
        return None

    @identifier.setter
    def identifier(self, idt):
        if not isinstance(idt, str):
            raise TypeError()
        type(self).__cache__[idt] = self
        self._identifier = idt

    @classmethod
    def from_rtype(cls, data, owner=None, identifier=None):
        """Instantiates a Spec from a ruamel commented collection."""
        if not isinstance(data, cls.__rtype__):
            msg = "Method call requires a ruamel Commented collection."
            raise TypeError(msg)
        if cls is SpecContainer:
            if isinstance(data, CommentedSeq):
                cls = SpecList
            elif isinstance(data, CommentedMap):
                cls = SpecDict
        instance = cls(owner=owner,
                       identifier=identifier,
                       _delay=True)
        instance._data = data
        instance.initialized = True
        instance.validate()
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
            return self.ispec.dump(self.ispec.__setter__(val))
        if isinstance(val, SpecContainer):
            return val._data
        else:
            return val

    @classmethod
    def load(cls, source, owner=None, identifier=None):
        """Loads source, either a file pointer, string or pathlib.Path."""
        return cls.from_rtype(cls._yaml.load(source),
                              owner=owner, identifier=identifier)

    def dump(self, sink):
        """Dumps spec to sink, either a file pointer or pathlib.Path."""
        return self._yaml.dump(self._data, sink)

    def json(self, sink=None):
        """Dumps a compact json to sink, either a file pointer or pathlib.Path.

        if sink is None, returns a json string (equivalent to json.dumps).
        """
        if sink:
            json.dump(self, sink, cls=SpecEncoder)
        else:
            return json.dumps(self, cls=SpecEncoder)

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

    def __repr__(self):
        idt = self.identifier
        if idt is not None:
            cls = type(self).__name__
            return '{cls}[{idt}]'.format(cls=cls, idt=idt)
        return super().__repr__()

    def __str__(self):
        stringio = io.StringIO()
        self.dump(stringio)
        return stringio.getvalue()


class SpecList(SpecContainer, MutableSequence):
    """An enumerated specification."""
    __rtype__ = CommentedSeq

    def __init__(self, iterable=(), owner=None, identifier=None,
                 _delay=False):
        self._data = CommentedSeq()
        self.extend(iterable)
        super().__init__(owner, identifier, _delay)

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
    # Either an attribute name or callable that returns the identifier.
    # If None, the identifer is set externally.
    __identifier__ = None

    def __init__(self, mapping=(), owner=None, identifier=None,
                 _delay=False, **kwargs):
        self._data = CommentedMap()
        self.update(mapping, **kwargs)
        # In case identifiers are internal, the argument is ignored silently.
        if self.__identifier__ is not None:
            identifier = None
        super().__init__(owner, identifier, _delay)

    def _post_init(self):
        super()._post_init()
        if type(self).__identifier__ is not None:
            type(self).__cache__[self.identifier] = self

    @property
    def specs(self):
        """The specification fields declared by the class."""
        return self.__keyspecs__

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

# =============================================================================
# Specification storage interface
# =============================================================================


class SpecStore(dts.StorageResource):
    """Base class for specification stores.

    The storage model is as follows: root/<owner>/<path>/<identifier>.yaml,
    where owner is an object that must print to a string which can be used
    in a file path, path is provided by the container type, and identity
    is an optional string argument.
    """

    @abc.abstractmethod
    def __store__(self, container, ownership, path, identifier):
        """The storage method defined by subclasses."""
        pass

    def store(self, container):
        """Stores a spec container."""
        if not isinstance(container, SpecContainer):
            raise TypeError()
        try:
            owner = container.owner
            ownership = str(owner)
            path = container.__path__
            identifier = container.identifier
            assert all((owner, path, identifier))
        except (AttributeError, TypeError, AssertionError):
            msg = "A spec container must have a valid owner, path and " + \
                  "identifier to be stored."
            raise ValueError(msg)
        self.__store__(container, ownership, path, identifier)

    @abc.abstractmethod
    def __retrieve__(self, ownership, path, identifier):
        """The retrieve method, which must return a valid source."""
        pass

    def retrieve(self, owner, path, identifier):
        try:
            ownership = str(owner)
            source = self.__retrieve__(ownership, path, identifier)
            cls = SpecContainerType.__path_registry__[path]
        except (TypeError, dts.ResourceError, KeyError):
            params = dict(owner=owner,
                          path=path,
                          identifier=identifier)
            msg = "Could not retrieve a spec sheet with parameters {}"
            raise dts.ResourceError(msg.format(params))
        return cls.load(source, owner, identifier)

    @abc.abstractmethod
    def __list__(self, ownership, path):
        """The list method, which must return a list of identifiers."""
        pass

    def list(self, owner, path):
        """List available specifications for owner on path."""
        ownership = str(owner)
        return self.__list__(ownership, path)

    @abc.abstractmethod
    def __delete_sheet__(self, ownership, path, identifier):
        """The deletion primitive for spec sheets."""
        pass

    def delete(self, owner, path, identifier, confirm=True):
        """Deletes a spec sheet."""
        params = dict(owner=owner,
                      path=path,
                      identifier=identifier)
        try:
            ownership = str(owner)
            SpecContainerType.__path_registry__[path]
        except (TypeError, KeyError):
            msg = "Could not retrieve a spec sheet with parameters {}"
            raise dts.ResourceError(msg.format(params))
        if confirm is not False:
            msg = "This will permanently delete a specification sheet " + \
                  "with these parameters {}. Would like to proceed? (Y/n)."
            confirmation = input(msg.format(params))
            if not confirmation == 'Y':
                msg = "Aborting deletion method."
                print(msg)
                return False
        try:
            self.__delete_sheet__(ownership, path, identifier)
            return True
        except dts.ResourceError:
            return False

    @abc.abstractmethod
    def __list_ownership__(self):
        """Primitive for listing ownership."""
        pass

    def list_ownership(self):
        """Returns a list of ownerships."""
        return self.__list_ownership__()

    @abc.abstractmethod
    def __drop_ownership__(self, ownership, force=False):
        """primitive for dropping owner."""
        pass

    def drop_owner(self, owner, confirm=True, force=False):
        """Drops owner from the specification store.

        Params:
            owner: the owner to be dropped.
            confirm: flag indicating whether to prompt the user.
            force: unless True, the action will fail if the store
                contains spec sheets for the owner.
        """
        if confirm is not False:
            msg = "The following owner: {} will be permanently deleted. " \
                "Do you wish to proceed? (Y/n)."
            confirmation = input(msg.format(owner))
            if not confirmation == 'Y':
                msg = "Aborting deletion method."
                print(msg)
                return False
        try:
            ownership = str(owner)
            self.__drop_ownership__(ownership, force=force)
            return True
        except (TypeError, dts.ResourceError):
            return False

    @abc.abstractmethod
    def __list_paths__(self, ownership):
        """Primitive for list_paths."""
        pass

    def list_paths(self, owner):
        """Lists implemented paths for supplied owner."""
        ownership = str(owner)
        return self.__list_paths__(ownership)

    @abc.abstractmethod
    def __drop_path__(self, path, ownership, force=False):
        """Primitive for drop_path."""
        pass

    def drop_path(self, path, confirm=True, force=False, owners=None):
        """Drops path for all owners from the specification store.

        Params:
            path: the path to be dropped.
            confirm: flag indicating whether to prompt the user.
            force: unless True, the action will fail on any path that
                contains specifications.
            owners: optional list of owners upon which to limit the action.
                Defaults to None, meaning that the action is carried out
                across the whole store.
        """
        if confirm is not False:
            if owners is None:
                scope = "all owners"
            else:
                scope = "owners: {}".format(owners)
            msg = "The following path: {0} will be permanently deleted " \
                "for {1}. Do you wish to proceed? (Y/n)."
            confirmation = input(msg.format(path, scope))
            if not confirmation == 'Y':
                msg = "Aborting deletion method."
                print(msg)
                return False
        if owners is None:
            ownership_list = self.list_ownership()
        else:
            ownership_list = [str(o) for o in owners]
        for ownership in ownership_list:
            self.__drop_path__(path, ownership, force=force)
        return True


class SpecDirectory(SpecStore):
    """A spec store in a locally accessible file system."""

    def __init__(self, path):
        self._root = Path(path)

    @property
    def root(self):
        return self._root

    def __exists__(self):
        return self._root.exists()

    def __empty__(self):
        return not os.listdir(self._root)

    def __create__(self):
        self._root.mkdir(parents=True, exist_ok=True)

    def __drop__(self, force=False):
        if force is True:
            shutil.rmtree(self._root)
        else:
            self._root.rmdir()

    def __store__(self, container, ownership, path, identifier):
        directory = self._root / os.path.join(ownership, path)
        if not directory.exists():
            directory.mkdir(parents=True)
        container.dump(directory / (identifier + '.yaml'))

    def __retrieve__(self, ownership, path, identifier):
        directory = self._root / os.path.join(ownership, path)
        file = directory / (identifier + '.yaml')
        if not file.exists():
            raise dts.ResourceError()
        return file

    def __list__(self, ownership, path):
        directory = self._root / os.path.join(ownership, path)
        paths = directory.glob('*.yaml')
        return [p.name[:-5] for p in paths]

    def __delete_sheet__(self, ownership, path, identifier):
        directory = self._root / os.path.join(ownership, path)
        file = directory / (identifier + '.yaml')
        try:
            os.remove(file)
        except FileNotFoundError:
            raise dts.ResourceError()

    def __list_ownership__(self):
        return [p.name for p in self._root.iterdir() if p.is_dir()]

    def __drop_ownership__(self, ownership, force=False):
        directory = self._root / ownership
        try:
            if force is True:
                shutil.rmtree(directory)
            else:
                directory.rmdir()
        except OSError:
            raise dts.ResourceError()

    def __list_paths__(self, ownership):
        directory = self._root / ownership
        return [p.name for p in directory.iterdir() if p.is_dir()]

    def __drop_path__(self, path, ownership, force=False):
        directory = self._root / os.path.join(ownership, path)
        try:
            if force is True:
                shutil.rmtree(directory)
            else:
                directory.rmdir()
        except OSError:
            raise dts.ResourceError()


class SpecBucketGCP(SpecStore):
    """A spec store in a Google Cloud Platform storage bucket.

    Unlike a directory-based specification store, bucket-based stores keep
    track of versions.

    Params:
        project: a cloud platform project name.
        path: a bucket name or path.
    """

    def __init__(self, project, path):
        self._client = Client(project)
        self._bucket = self._client.bucket(path)

    @property
    def client(self):
        return self._client

    @property
    def bucket(self):
        return self._bucket

    def __exists__(self):
        if not self._bucket.exists():
            return False
        if not self._bucket.versioning_enabled:
            msg = "A bucket-based specification store should enable " + \
                  "versions, however {0} does not."
            self.warn(msg.format(self))
        return True

    def __empty__(self):
        itr = self._bucket.list_blobs()
        try:
            next(iter(itr))
        except StopIteration:
            return True
        else:
            return False

    def __create__(self):
        if self.exists():
            self.__drop__()
        self._bucket.create()
        self._bucket.versioning_enabled = True
        self._bucket.patch()

    def __drop__(self, force=False):
        if force is True:
            blobs = self._bucket.list_blobs(versions=True)
            self._bucket.delete_blobs(list(blobs))
        time.sleep(1)
        self._bucket.delete()

    def blob(self, ownership, path, identifier):
        name = '/'.join([ownership, path, identifier + '.yaml'])
        return self._bucket.blob(name)

    def __store__(self, container, ownership, path, identifier):
        blob = self.blob(ownership, path, identifier)
        blob.upload_from_string(str(container))

    def __retrieve__(self, ownership, path, identifier):
        blob = self.blob(ownership, path, identifier)
        try:
            return blob.download_as_string().decode('utf-8')
        except NotFound:
            raise dts.ResourceError()

    def __list__(self, ownership, path):
        prefix = '/'.join([ownership, path])
        blobs = self._bucket.list_blobs(prefix=prefix)
        return [b.name.split('/')[-1][:-5] for b in blobs]

    def __delete_sheet__(self, ownership, path, identifier):
        blob = self.blob(ownership, path, identifier)
        try:
            blob.delete()
        except NotFound:
            raise dts.ResourceError()

    def __list_ownership__(self):
        def ownership(blob):
            return blob.name.split('/')[0]
        return list(set(ownership(b) for b in self._bucket.list_blobs()))

    def __drop_ownership__(self, ownership, force=False):
        if not force:
            return
        prefix = ownership
        blobs = self._bucket.list_blobs(prefix=prefix)

        def delete():
            for b in blobs:
                try:
                    b.delete()
                except:
                    continue

        deletion_thread = Thread(target=delete)
        deletion_thread.start()

    def __list_paths__(self, ownership):
        prefix = ownership

        def path(blob):
            return '/'.join(blob.name.split('/')[1:-1])
        paths = (path(b) for b in self._bucket.list_blobs(prefix=prefix))
        return list(set(paths))

    def __drop_path__(self, path, ownership, force=False):
        if not force:
            return
        prefix = '/'.join([ownership, path])
        blobs = self._bucket.list_blobs(prefix=prefix)

        def delete():
            for b in blobs:
                try:
                    b.delete()
                except:
                    continue

        deletion_thread = Thread(target=delete)
        deletion_thread.start()
