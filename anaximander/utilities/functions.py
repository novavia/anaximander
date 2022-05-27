"""Various general-purpose utility functions."""

# =============================================================================
# Imports and constants
# =============================================================================

import abc
import collections
from collections.abc import MutableMapping, MutableSequence, MutableSet
from contextlib import contextmanager
from dataclasses import dataclass, field
from frozendict import frozendict
from functools import wraps
import functools
import inspect
import re
import socket
import time
import types
from typing import Any, Callable, Collection, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd
import petl as etl

Namespace = dict[str, Any]
Filter = Callable[[Any], bool]

# =============================================================================
# Functions
# =============================================================================


def boolean(string):
    """Converts a string to a boolean."""
    if string == "True":
        return True
    elif string == "False":
        return False
    else:
        raise ValueError


def is_online():
    """Function that determines if the tester is online."""
    host = socket.gethostbyname("www.google.com")
    try:
        my_socket = socket.create_connection((host, 80), 2)
    except Exception:
        return False
    else:
        my_socket.close()
        return True


def indent(string, spacing=4):
    white = " " * spacing
    return white + white.join(string.splitlines(keepends=True))


def unindent(string, spacing=4):
    white = " " * spacing

    def unindent_line(line):
        if line[:spacing] != white:
            msg = f"Cannot unindent {line} with {spacing} spaces."
            raise ValueError(msg)
        return line[spacing:]

    return "".join(unindent_line(li) for li in string.splitlines(keepends=True))


def multistring(string, level=0):
    """Allows to specify an indented multistring for reading clarity."""
    lines = string.splitlines(True)
    if not lines:
        return string
    if lines[-1].strip() == "":
        lines = lines[1:-1]
    else:
        lines = lines[1:]
    return unindent("".join(lines), level * 4)


def camel_to_snake(string: str) -> str:
    """Transforms a CamelCase string to a matching snake_case string."""
    string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", string)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", string).lower()


def snake_to_camel(string: str) -> str:
    """Transforms a snake_case string to a matching CamelCase string."""
    return "".join(word.title() for word in string.split("_"))


def auto_label(collection: Iterable, *, separator="-", label_first=False):
    """Adds sequential integers to items for representation purposes."""
    threshold = 0 if label_first else 1
    return [
        (str(item) + separator + str(i + 1) if i >= threshold else str(item))
        for i, item in enumerate(collection)
    ]


def sort_by(source: Iterable, target: Iterable):
    source, target = list(source), list(target)
    source_index, target_index = source.index, target.index
    target_length = len(target)

    def sort_key(item):
        try:
            return target_index(item)
        except ValueError:
            return target_length + source_index(item)

    return sorted(source, key=sort_key)


def get_attr_if_exist(obj: Any, name: str):
    return getattr(obj, name, None)


def as_dict(obj, properties_default_to_none=True):
    """Returns a mapping of object's attributes and properties.

    Includes attributes, class attributes and properties.
    Since property invocation can fail, the function will by default set
    the corresponding dictionary attribute to None. The behavior can be
    overridden to raise errors by setting the flag to False.
    """
    dictionary = {}
    if properties_default_to_none:
        attr_getter = get_attr_if_exist
    else:
        attr_getter = getattr
    for name in dir(obj):
        if name.startswith("__"):
            continue
        attr = attr_getter(obj, name)
        if inspect.ismethod(attr):
            continue
        dictionary[name] = attr
    return dictionary


def list_of_dicts_to_table(data: list[dict]) -> dict[Any, list]:
    dataframe = pd.DataFrame(data).replace({np.nan: None})
    return etl.fromdataframe(dataframe)


class LookupType(abc.ABCMeta):
    """A metaclass for streamlining subclass registration & lookup.

    The key is the attribute name that serves to register subclasses.
    The implementation requires a reentrant lock assisgned to cls.__rlock__.
    """

    def _register_type(cls, subclass):
        try:
            with cls.__rlock__:
                key = getattr(subclass, cls.__key__)
                cls.__lookup__[key] = subclass
        except AttributeError:
            pass

    def _lookup_type(cls, key):
        with cls.__rlock__:
            return cls.__lookup__[key]

    @abc.abstractstaticmethod
    def _type_factory(key):
        return NotImplemented

    def __getitem__(cls, key):
        with cls.__rlock__:
            try:
                return cls.__lookup__[key]
            except KeyError:
                new_type = cls._type_factory(key)
                return new_type


def namespace(
    *filters: Filter,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    _secondary=False,
) -> Namespace:
    """A wrapper around 'locals' that captures the current frame's namespace.

    :*filters: an iterable of filter functions
    :param include: if provided, only keys found in include are returned
    :param exclude: if provided, no key found in exclude is returned
    :param _secondary: system attribute used by derivative functions
    """
    caller_frame = inspect.currentframe().f_back
    target_frame = caller_frame.f_back if _secondary else caller_frame
    candidates = target_frame.f_locals.copy()
    # We remove hidden attributes and special attributes (@ is introduced by pytest)
    candidates = {k: v for k, v in candidates.items() if not k[0] in (["@", "_"])}
    if include is not None:
        candidates = {k: v for k, v in candidates.items() if k in include}
    if exclude is not None:
        candidates = {k: v for k, v in candidates.items() if k not in exclude}
    if filters:
        for filter in filters:
            candidates = {k: v for k, v in candidates.items() if filter(v)}
    return candidates


def method_arguments(
    _ref: Optional[str] = None,
    *filters: Filter,
    unpack: Union[bool, str] = True,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Namespace:
    """As the first line to a method's code, returns supplied method arguments.

    :param _ref: an optional string to remove from locals. By default, the
        function pops 'self' and 'cls'. If supplied, ref is also popped.
    :*filters: an iterable of filter functions
    :param unpack: determines whether to unpack keyword arguments. Assumes
        keyword arguments are to be found in the 'kwargs' local variable, or
        the string supplied to unpack.
    :param include: if provided, only keys found in include are returned
    :param exclude: if provided, no key found in exclude is returned
    """
    exclude = set() if exclude is None else set(exclude)
    exclude |= {"self", "cls"}
    if _ref is not None:
        exclude.add(_ref)
    arguments = namespace(*filters, include=include, exclude=exclude, _secondary=True)
    if unpack is False:
        return arguments
    elif unpack is True:
        kv_arg = "kwargs"
    elif isinstance(unpack, str):
        kv_arg = unpack
    else:
        raise TypeError("unpack must be bool or str")
    kwargs = arguments.pop(kv_arg, {})
    return arguments | kwargs


def named_arguments(callable: Callable) -> list[str]:
    """Returns a tuple of callable's named arguments."""
    args = []
    signature = inspect.signature(callable)
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        args.append(param.name)
    return args


def mandatory_arguments(callable: Callable) -> list[str]:
    """Returns the arguments that don't defined default values and must be supplied."""
    mandatory = []
    signature = inspect.signature(callable)
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty:
            mandatory.append(param.name)
    return mandatory


def takes_keyword_arguments(callabe: Callable) -> bool:
    """True if callable accepts variable keyword arguments."""
    signature = inspect.signature(callable)
    for param in signature.parameters.values():
        if param.kind is param.VAR_KEYWORD:
            return True
    return False


def attribute_owner(cls: type, name: str) -> type:
    """Returns cls' mro member that defines name, or raises AttributeError."""
    for c in cls.__mro__:
        if name in c.__dict__:
            return c
    else:
        raise AttributeError


class Singleton(type):
    """A metaclass for making singleton types."""

    __basetype__: type = None
    _instance: "Singleton" = None

    def __new_method__(cls):
        new_method = cls.__new__

        @wraps(new_method)
        def wrapped(cls, *args, **kwargs):
            if not "_instance" in vars(cls):
                cls._instance = instance = object.__new__(cls)
                dict_ = cls.__basetype__(*args, **kwargs).__dict__
                instance.__dict__ = dict_.copy()
                return instance
            return cls._instance

        return wrapped

    def __init__(cls, name, bases, namespace):
        cls.__basetype__ = bases[0]
        cls.__new__ = cls.__new_method__()


def _singleton(cls: type, *args, **kwargs) -> Singleton:
    """Transforms a class to make it a singleton."""
    base_metaclass = type(cls)
    if issubclass(base_metaclass, Singleton):
        metaclass = base_metaclass
    elif not base_metaclass is type:
        name = base_metaclass.__name__ + "Singleton"
        metaclass = types.new_class(name, (Singleton, base_metaclass))
    else:
        metaclass = Singleton
    new_cls = types.new_class(cls.__name__, (cls,), kwds={"metaclass": metaclass})
    return new_cls(*args, **kwargs)


def singleton(*args, **kwargs) -> Singleton:
    """Decorates a class with optional arguments to return a singleton.

    The arguments are passed to the decorated class to make the one and only
    instance. The decorator returns that instance, not the class.
    """
    if not kwargs and len(args) == 1 and isinstance(args[0], type):
        # If these conditions are met, we assume the decorator is written
        # with no arguments. There is an edge case where this assumption
        # could break, and we'll live with it! :-)
        return _singleton(args[0])
    return lambda cls: _singleton(cls, *args, **kwargs)


class typeproperty(property):
    """A metaclass property that allows regular properties of the same name."""

    def __set__(self, cls, value):
        if isinstance(value, property):
            # In order to write to cls, the property must be temporarily
            # removed from its metaclass -which forces a recursive search
            # in case the property is declared in a parent metaclass
            name = self.fget.__name__
            metaclass_owner = attribute_owner(type(cls), name)
            delattr(metaclass_owner, name)
            setattr(cls, name, value)
            # We then reinstate self as a metaclass descriptor
            setattr(metaclass_owner, name, self)
            return
        return super().__set__(cls, value)


def inherited_property(prop: typeproperty):
    """Makes an instance property that reads its value from a type property."""
    name = prop.fget.__name__
    return property(lambda obj: getattr(type(obj), name))


@dataclass
class ProcessTimer:
    name: Optional[str] = None
    records: list[float] = field(default_factory=list, repr=False)

    def punch(self):
        self.records.append(time.process_time())

    def __enter__(self) -> None:
        self.records = []
        self.punch()
        return self

    def __exit__(self, *exc) -> float:
        self.punch()
        if len(self.records) > 2:
            laps = [
                round(1e3 * (t1 - t0))
                for t0, t1 in zip(self.records[:-1], self.records[1:])
            ]
        else:
            laps = []
        proc_time = round(1e3 * (self.records[-1] - self.records[0]))
        naming = f" for {self.name}" if self.name else ""
        msg = f"Process time{naming} is {proc_time} ms"
        print(msg)
        if laps:
            for i, lap in enumerate(laps):
                i += 1
                msg = f"Step {i}: {lap} ms"
                print(msg)


@contextmanager
def process_timer(name: str, **kwargs):
    """Functional form for ProcessTimer."""
    timer = ProcessTimer(name)
    try:
        timer.__enter__()
        yield timer
    finally:
        timer.__exit__()


def print_process_time(func, **kwargs):
    """A function proces time decorator."""
    timer_ = ProcessTimer(func.__name__, **kwargs)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with timer_:
            return func(*args, **kwargs)

    return wrapped


def freeze(collection: Collection):
    """Deeply freezes basic collection types to make a hashable object."""
    if not isinstance(collection, Collection):
        return collection
    if isinstance(collection, MutableMapping):
        return frozendict({k: freeze(v) for k, v in collection.items()})
    elif isinstance(collection, MutableSequence):
        return tuple(freeze(v) for v in collection)
    elif isinstance(collection, MutableSet):
        return frozenset(freeze(v) for v in set)
