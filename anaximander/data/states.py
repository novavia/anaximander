#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines states.

States are defined as classes, whose instances evaluate equal as long
as their attributes are equal.

In order to distinguish between types of states and concrete states
defined as classes, we separate classes between ConcreteState subclasses
and AbstractState subclasses.

For instance, the activity of a machine that can be either operating or
idle can be declared with an ActivityState class that is made abstract,
and two derived types OperatingState and IdleState. This is signaled by
using the keyword abstract in the class declaration keyword arguments.

ConcreteState classes have a label, which is used as shorthand to designate
the corresponding state. It also serves to register the ConcreteState class
with its AbstractState parents (there can be multiple if the state types
form a hierarchy). ConcreteState classes also define a default color that
is used for plotting.

While State classes can be chained through inheritance, it is forbidden
to make an AbstractState class inherit from a ConcreteState class. Further,
multiple inheritance of State classes is forbidden as well.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
from collections import ChainMap

# =============================================================================
# Base classes
# =============================================================================


class StateType(abc.ABCMeta):
    """Metaclass for states."""

    def __new__(mcl, name, bases, namespace, abstract=False):
        # Cannot pass multiple State bases
        if sum(isinstance(b, mcl) for b in bases) > 1:
            raise TypeError
        if abstract is None:  # this one of the base classes
            pass
        else:
            concrete = any(issubclass(b, ConcreteState) for b in bases)
            if abstract is True:
                # Check that ConcreteState is not on the inheritance path
                if concrete:
                    raise TypeError
            elif abstract is False:
                if not concrete:
                    bases = (ConcreteState,) + bases
            else:
                raise ValueError  # abstract should be True or False
        return super().__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace, abstract=False):
        if abstract is None:
            return
        elif abstract is True:
            abstract_bases = [b for b in bases if issubclass(b, AbstractState)]
            abc_cls = abstract_bases[0]
            base_states = abc_cls._states
            cls._registry = {}
            cls._states = ChainMap(cls._registry, base_states)
        elif abstract is False:
            abstract_parents = [c for c in cls.mro() if
                                issubclass(c, AbstractState)]
            abc_cls = abstract_parents[0]  # Guaranteed to exist
            abc_cls._register(cls)

    @property
    def labels(cls):
        if issubclass(cls, ConcreteState):
            return NotImplemented
        else:
            return set(cls._states)

    @property
    def states(cls):
        if issubclass(cls, ConcreteState):
            return NotImplemented
        else:
            return set(cls._states.values())


class ConcreteState(metaclass=StateType, abstract=None):

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls, *args, **kwargs)


class AbstractState(metaclass=StateType, abstract=None):
    _states = {}  # Registry seed

    def __new__(cls, *args, **kwargs):
        if args:
            label = args[0]
            args = args[1:]
        else:
            label = kwargs.pop('label', None)
        try:
            cls = cls._registry[label]
            return cls(*args, **kwargs)
        except KeyError:
            msg = f"Cannot instantiate Abstract State {cls.__name__}"
            msg += " without a registered state."
            raise TypeError(msg)

    @classmethod
    def _register(cls, state_class):
        cls._registry[state_class.label] = state_class


class State(AbstractState, metaclass=StateType, abstract=True):
    label = None
    color = None

    def __eq__(self, other):
        if type(self) == type(other):
            return self.__dict__ == other.__dict__
        return False

    def __repr__(self):
        return f"<{type(self).__name__}>"


class UndefinedState(State):
    label = 'undefined'
    color = 'gray'
