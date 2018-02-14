#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to Anaximander's meta package.

One of the most salient characteristics of this package is the archetype
model. Archetypes are an implementation of generic types, comparable to
C++ templates. Hence archetypes are abstract base types, whose derived
concrete types are created by supplying them with parameter values. These
parameters are called metacharacters. As a dummy example, assume you
create a nested list archetype, such that for a given nested depth there
exists one concrete class. Conceptually, NestedList[0] is a class holding
scalars, NestedList[1] is similar to a simple list, NestedList[2] is
a list of lists, and so on. Here NestedList is declared as an archetype,
with a single metacharacter that could be called depth. The derived types
may either be declared explicitly through a class declaration that specifies
the value given to the metacharacter as in:

    class NestedList2(NestedList, depth=2):
        ...

Or they can also be created programatically -and even implicitly, by simply
stating NestedList[2].

Generally, the explicit declarative style is preferred when the archetypical
features are not sufficient to fully describe the behavior of the derived
types. It also possible to further specialize a subtype of an archetype.

The interface for creating archetypes is simple: decorate a class with
the @archetype decorator to make it an archetype. The metacharacters
are declared in the class declaration, as if they were regular descriptors,
even though they then technically become metaclass-level descriptors.

The mechanics of archetype creation are not exactly trivial. Here is a short
description to satsify the reader's curiosity and shed some light on the code
contained in this package:

    * Because archetypes are specialized abstract base classes, they must
    have a metaclass that defines them as archetypes;
    * However, this dictates that the concrete types that are derived from an
    archetype cannot have the same metaclass;
    * We resolve this by refering to the archetype-decorated class declaration
    as the basetype. The decorator returns a different class that is the
    archetype.
    * Derived types inherit from the basetype. They still appear to inherit
    from the archetype (i.e. issubclass and isinstance will work as expected),
    but the archetype is in fact a dead end in the class hierarchy.
    * To confer its properties to the archetype, the decorator creates
    a one-time use metaclass mixin. This metaclass mixin is a subclass
    of the base ArcheType mixin metaclass.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""


class NxMetaError(Exception):
    """Customized error for metaprogramming errors."""
    pass


from .nxdescriptors import *
#from .nxmetas import *
#from .nxtypes import *
