#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxdescriptors.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

from itertools import count

import pytest

from anaximander3.meta import nxdescriptors as nxd

# =============================================================================
# Test Cases
# =============================================================================


class Type(type):
    _id_counter = count()

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls.class_id = next(cls._id_counter)
        nxd.NxDescriptor.collect(cls, namespace)

    @property
    def classname(cls):
        return cls.__name__


class C(metaclass=Type):
    x: int = nxd.NxAttribute()
    classname: str = nxd.TypeAttributeProperty()


class D(metaclass=Type):
    class_id: int = nxd.TypeAttributeProperty()


def test_instantiation():
    assert C.x.name is 'x'
    with pytest.raises(AttributeError):
        C.x.name = 'x'
    assert repr(C.x) == '<NxAttribute name:x>'
    assert list(C.__nxdescriptors__) == ['x', 'classname']


def test_objecttypeproperty():
    assert C.classname == 'C'
    c = C()
    assert c.classname == 'C'


class X(metaclass=Type):
    a = nxd.NxAttribute()
    b = nxd.NxAttribute()
    classname = nxd.TypeAttributeProperty()


class Y(metaclass=Type):
    c = nxd.NxAttribute()
    d = nxd.NxAttribute()


def test_collect():
    """Tests the proper collection of descriptors in sequence."""
    class Z(X, Y):
        e = nxd.NxAttribute()

    assert list(Z.__nxdescriptors__) == ['a', 'b', 'classname', 'e', 'c', 'd']
    assert Z.classname == 'Z'

    class Z(X, Y):
        e = nxd.NxAttribute()
        a = nxd.NxAttribute()
        c = nxd.NxAttribute()
        b = nxd.NxAttribute()
        classname = nxd.TypeAttributeProperty()

    assert list(Z.__nxdescriptors__) == ['e', 'a', 'c', 'b', 'classname', 'd']
    assert Z.a is not X.a
    assert Z.classname == 'Z'
    assert X.classname == 'X'
    assert Z.__nxdescriptors__['classname'] is not \
        X.__nxdescriptors__['classname']

    with pytest.raises(nxd.NxMetaError):
        class Z(X, Y):
            z = nxd.NxAttribute()
            b = nxd.NxAttribute()
            a = nxd.NxAttribute()

    with pytest.raises(nxd.NxMetaError):
        class Z(X, Y):
            b = nxd.NxAttribute()


def test_nxattributes():

    class K(metaclass=Type):
        x = nxd.NxAttribute(default='x', type_=str)
        y: int = nxd.NxAttribute(nullable=False, validate=lambda v: v > 0)
        y2: int = nxd.NxAttribute(nullable=False)
        z: int = nxd.NxAttribute(nullable=True, set_once=True,
                                 type_=str, cache='__z')
        c = nxd.NxAttribute(default=nxd.call(list))
    
        @y2.validator
        def validate_y2(self, attr, value):
            return value > 0

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
    with pytest.raises(ValueError):
        k.y2 = -1
    with pytest.raises(TypeError):
        k.y2 = None
    k.y2 = 1
    k.z = None
    assert k.z is None
    assert k.__z is None
    with pytest.raises(AttributeError):
        k.z = 0
    assert k.c == []


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
