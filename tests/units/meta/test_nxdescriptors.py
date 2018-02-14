#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for xprops.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

from itertools import count

import pytest

from anaximander2.meta import nxdescriptors as nxd

# =============================================================================
# Test Cases
# =============================================================================


class Type(type):
    _id_counter = count()

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls.class_id = next(cls._id_counter)
        for k, v in namespace.items():
            if isinstance(v, nxd.NxDescriptor):
                v.bind(k, cls)
        cls.__nxdescriptors__ = nxd.ObjectDescriptor.collect(bases, namespace)

    @property
    def classname(cls):
        return cls.__name__


class C(metaclass=Type):
    x = nxd.ObjectCharacter()
    classname = nxd.ObjectTypeProperty()


class D(metaclass=Type):
    class_id = nxd.ObjectTypeProperty()


def test_instantiation():
    assert C.x.name is 'x'
    with pytest.raises(AttributeError):
        C.x.name = 'x'
    assert repr(C.x) == '<ObjectCharacter name:x>'
    assert list(C.__nxdescriptors__) == ['x', 'classname']


def test_objecttypeproperty():
    assert C.classname == 'C'
    c = C()
    assert c.classname == 'C'


class X(metaclass=Type):
    a = nxd.ObjectCharacter()
    b = nxd.ObjectCharacter()
    classname = nxd.ObjectTypeProperty()


class Y(metaclass=Type):
    c = nxd.ObjectCharacter()
    d = nxd.ObjectCharacter()


def test_collect():
    """Tests the proper collection of descriptors in sequence."""
    class Z(X, Y):
        e = nxd.ObjectCharacter()

    assert list(Z.__nxdescriptors__) == ['a', 'b', 'classname', 'e', 'c', 'd']
    assert Z.classname == 'Z'

    class Z(X, Y):
        e = nxd.ObjectCharacter()
        a = nxd.ObjectCharacter()
        c = nxd.ObjectCharacter()
        b = nxd.ObjectCharacter()
        classname = nxd.ObjectTypeProperty()

    assert list(Z.__nxdescriptors__) == ['e', 'a', 'c', 'b', 'classname', 'd']
    assert Z.a is not X.a
    assert Z.classname == 'Z'
    assert X.classname == 'X'
    assert Z.__nxdescriptors__['classname'] is not \
        X.__nxdescriptors__['classname']

    with pytest.raises(nxd.NxMetaError):
        class Z(X, Y):
            z = nxd.ObjectCharacter()
            b = nxd.ObjectCharacter()
            a = nxd.ObjectCharacter()

    with pytest.raises(nxd.NxMetaError):
        class Z(X, Y):
            b = nxd.ObjectCharacter()


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
