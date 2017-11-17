#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of general-purpose utility functions.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

from collections import deque
import functools
import itertools
import socket
import sys
import os
import subprocess

# =============================================================================
# Attributes handling
# =============================================================================


def get(val, default=None):
    """This function returns val if not None, else default."""
    return val if val is not None else default


def lmap(*args):
    """Builds a dictionary from locally named variables.

    If a variable x is defined in the caller's frame, and x is passed
    to lmap, the function returns {'x': x}.

    :param *args: a sequence of local variables names.
    :return: a dictionary of locally named variables.
    """
    caller_locals = sys._getframe(1).f_locals
    return {k: v for k, v in caller_locals.items() if k in args}


def typecheck(*types):
    """Returns a callable equivalent to isinstance."""
    def checker(obj):
        return isinstance(obj, types)
    return checker


def subcheck(*types):
    """Returns a callable equivalent to issubclass."""
    def checker(cls):
        return issubclass(cls, types)
    return checker


def boolean(string):
    """Converts a string to a boolean."""
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise ValueError

# =============================================================================
# String formatting
# =============================================================================


def spformat(obj, singular='item', plural=None):
    """Returns 'n {item}' or 'n {items}' where n is a collection's length.

    :param obj: a collection-like object (defines __len__) or an integer.
    :param singular: the naming of an item, defaults to 'item'.
    :param plural: plural form, defaulting to singular + 's'.
    :return: formatted string.
    """
    plural = plural or singular + 's'
    n = obj if isinstance(obj, int) else len(obj)
    if n <= 1:
        return '{n} {sf}'.format(n=n, sf=singular)
    else:
        return '{n} {pf}'.format(n=n, pf=plural)


class curlydict(dict):
    """A dictionary that returns '{k}' when passed a missing string key k."""

    def __missing__(self, key):
        return '{' + key + '}'


# XXX: this is made somewhat obsolete by python 3.6's format strings.
def lformat(string):
    """Applies format to string from the caller's local dictionary.

    For instance, if x is defined and equal to 3,
    lformat('x is {x}') returns 'x is 3'.
    """
    caller_locals = sys._getframe(1).f_locals
    return string.format(**curlydict(caller_locals))


def iformat(*attrs):
    """Returns standard instance formatter with supplied attributes."""
    def formatter(inst):
        cls = type(inst).__name__
        if not attrs:
            return f"<{cls}>"
        keyvals = {}
        for a in attrs:
            path = a.split('.')
            obj = inst
            for link in path[:-1]:
                obj = getattr(obj, link)
            keyvals[path[0]] = getattr(obj, path[-1])
        content = " ".join(f"{k}:{v}" for k, v in keyvals.items())
        return f"<{cls} {content}>"
    return formatter


def nxrepr(*attrs):
    """A class decorator that adds standard representation."""
    def decorator(cls):
        cls.__repr__ = iformat(*attrs)
        return cls
    return decorator

# =============================================================================
# Iteration / collection functions
# =============================================================================


def dictunion(*dicts, unique_keys=False):
    """Returns a dictionary that unionizes the supplied dictionaries.

    If unique_keys is True, then the function raises a ValueError if
    the same key is featured in multiple input dictionaries.
    """
    union = {k: v for d in dicts for k, v in d.items()}
    if unique_keys:
        if len(union) != sum(len(d) for d in dicts):
            msg = "Dictionaries with shared keys were passed to dictunion " + \
                  "with the unique_keys flag set to True."
            raise ValueError(msg)
    return union


def vfilter(function, mapping):
    """Analogous to built-in filter, but targets a mapping's values.

    Returns an iterator of key, value pairs where the values have been
    filtered by function returning True.
    """
    return ((k, v) for k, v in mapping.items() if function(v))


def kfilter(function, mapping):
    """Analogous to built-in filter, but targets a mapping's keys.

    Returns an iterator of key, value pairs where the keys have been
    filtered by function returning True.
    """
    return ((k, v) for k, v in mapping.items() if function(k))


def vmap(function, mapping):
    """Analogous to built-in map, but targets a mapping's values.

    Returns an iterator of key, value pairs where function has been
    applied to values.
    """
    return ((k, function(v)) for k, v in mapping.items())


def pairwise(iterable, step=1):
    """Returns an iterator of element pairs, progressing with step."""
    left, right = itertools.tee(iterable)
    next(right, None)
    pairs = zip(left, right)
    if step == 1:
        return pairs
    else:
        return itertools.islice(pairs, 0, None, step)

# =============================================================================
# Function decorators
# =============================================================================


def args_or_kwargs(f=None, *, tabs=1):
    """Enforces calls to use either *args or **kwargs but not both.

    :param f: the decorated function.
    :param tabs: specifies any number of non-keyword arguments that escape
    the rule. The default is 1, which means that the decorator can be used
    without arguments for a method written as f(self, *args, **kwargs).
    """
    if f is None:
        return functools.partial(args_or_kwargs, tabs=tabs)
    err_msg = "{} can be called with *args or **kwargs but not both."
    msg = err_msg.format(f.__name__)

    @functools.wraps(f)
    def decorated(*arguments, **kwargs):
        if arguments[tabs:] and kwargs:
            raise ValueError(msg)
        return f(*arguments, **kwargs)
    return decorated


def passthrough(*types):
    """Passes through first arg to decorated function if it is in types."""
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            try:
                if isinstance(args[0], types):
                    return args[0]
            except IndexError:
                pass
            return f(*args, **kwargs)
        return decorated
    return decorator

# =============================================================================
# Metaprogramming
# =============================================================================


def meta(obj):
    """Returns the metatype of an object."""
    return type(type(obj))


def metargs(cls):
    """Returns the name, bases and namespace of the supplied class."""
    return (cls.__name__, cls.__bases__, cls.__dict__.copy())


def monkeypatch(cls, mixin, *exclusions):
    """Adds attributes and methods from mixin to cls.

    Attributes to exclude can be passed to exclusions as strings.
    Note that the patching is non-recursive, i.e. only attributes and
    methods declared in mixin are patched. If mixin inherits from a base
    class, then the attributes of the base class will not be patched.
    """
    # Reserved attributes that don't get overwritten
    xattrs = {'__module__', '__dict__', '__weakref__', '__doc__',
              '__patches__'}
    xattrs.update(exclusions)
    for name, attr in mixin.__dict__.items():
        if name not in xattrs:
            setattr(cls, name, attr)
    if not hasattr(cls, '__patches__'):
        cls.__patches__ = deque()
    cls.__patches__.appendleft(mixin)

# =============================================================================
# Networking
# =============================================================================


def is_online():
    """Function that determines if the tester is online."""
    try:
        host = socket.gethostbyname("www.google.com")
        socket.create_connection((host, 80), 2)
    except:
        return False
    else:
        return True


def get_gcloud_config(attr):
    """Get gcloud configuration value.

    This function wraps a subprocess call to "$gcloud config get-value".
    
    :param attr: a string, the name of the configuration attribute to retrieve.
                 For a complete list of valid input, refer to 
                 $gcloud config get-value --help
    :return:     a string, the configuration parameter value
    """
    if not os.getenv("LOCAL", False):
        msg = "gcloud config is only available on a local environment"
        raise Exception(msg)
    output = subprocess.check_output(["gcloud", "config", "get-value", attr])
    if not output:
        return ""
    return output.decode().replace("\n", "")


def get_gcloud_project_id():
    """Get Google Cloud current project id."""
    is_local = (os.getenv("LOCAL", True) == "True")
    if is_local:
        return get_gcloud_config("project")
    else:
        return os.getenv("GCLOUD_PROJECT", "")