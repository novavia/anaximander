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

from collections import OrderedDict
from itertools import count

import pytest

from anaximander2.meta import nxdescriptors as nxd

# =============================================================================
# Test Cases
# =============================================================================


class Type(type):
    class_id = nxd.MetaCharacter(name='class_id')
    _id_counter = count()

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls.__nxdescriptors__ = OrderedDict()
        cls.class_id = next(cls._id_counter)
        for k, v in namespace.items():
            if isinstance(v, nxd.NxDescriptor):
                v.name = k
                v.cls = cls
                v.register(cls)

    @property
    def classname(cls):
        return cls.__name__


class C(metaclass=Type):
    x = nxd.NxDescriptor()
    classname = nxd.TypeProperty()
    y = nxd.MetaCharacter()


class D(metaclass=Type):
    class_id = nxd.TypeProperty()


def test_instantiation():
    assert C.x.name is 'x'
    with pytest.raises(AttributeError):
        C.x.name = 'x'
    assert repr(C.x) == '<NxDescriptor name:x>'
    assert list(C.__nxdescriptors__) == ['x', 'classname', 'y']


def test_typeproperty():
    assert C.classname == 'C'
    c = C()
    assert c.classname == 'C'


def test_metacharacter():
    assert C.class_id == 0
    assert D.class_id == 1
    c, d = C(), D()
    with pytest.raises(TypeError):
        c.y
    assert d.class_id == 1
    # Type instances don't support inheritance because it would force
    # overriding a metacharacter.
    with pytest.raises(AttributeError):
        class E(C):
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
