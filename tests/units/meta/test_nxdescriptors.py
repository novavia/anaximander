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
        nxd.ObjectDescriptor.collect(cls, namespace)

    @property
    def classname(cls):
        return cls.__name__


class C(metaclass=Type):
    x: int = nxd.ObjectCharacter()
    classname: str = nxd.ObjectTypeProperty()


class D(metaclass=Type):
    class_id: int = nxd.ObjectTypeProperty()


def test_instantiation():
    assert C.x.name is 'x'
    with pytest.raises(AttributeError):
        C.x.name = 'x'
    assert repr(C.x) == '<ObjectCharacter name:x>'
    assert list(C.__objectdescriptors__) == ['x', 'classname']


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

    assert list(Z.__objectdescriptors__) == ['a', 'b', 'classname',
                                             'e', 'c', 'd']
    assert Z.classname == 'Z'

    class Z(X, Y):
        e = nxd.ObjectCharacter()
        a = nxd.ObjectCharacter()
        c = nxd.ObjectCharacter()
        b = nxd.ObjectCharacter()
        classname = nxd.ObjectTypeProperty()

    assert list(Z.__objectdescriptors__) == ['e', 'a', 'c', 'b',
                                             'classname', 'd']
    assert Z.a is not X.a
    assert Z.classname == 'Z'
    assert X.classname == 'X'
    assert Z.__objectdescriptors__['classname'] is not \
        X.__objectdescriptors__['classname']

    with pytest.raises(nxd.NxMetaError):
        class Z(X, Y):
            z = nxd.ObjectCharacter()
            b = nxd.ObjectCharacter()
            a = nxd.ObjectCharacter()

    with pytest.raises(nxd.NxMetaError):
        class Z(X, Y):
            b = nxd.ObjectCharacter()


def test_protected_attributes():

    class K(metaclass=Type):
        x = nxd.ObjectAttribute(default='x', type=str)
        y: int = nxd.ObjectCharacter(nullable=False, validate=lambda v: v > 0)
        z: int = nxd.ObjectCharacter(nullable=True, type=str, cache='__z')

    k = K()
    assert k.x == 'x'
    k.x = 'hey'
    assert k.x == k._x == 'hey'
    k.x = 'ho'
    assert k.x == k._x == 'ho'
    with pytest.raises(TypeError):
        k.x = 1
    k = K()
    with pytest.raises(ValueError):
        k.y = -1
    with pytest.raises(TypeError):
        k.y = None
    k.y = 1
    k.z = None
    assert k.z is None
    assert k.__z is None
    with pytest.raises(AttributeError):
        k.z = 0


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
