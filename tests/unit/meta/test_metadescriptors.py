#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxtype.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

import anaximander.utilities.functions as fun
import anaximander.meta.metadescriptors as mtd
from anaximander.meta.nxtype import nxtype, archetype, prototype
from anaximander.meta.nxobject import NxObject

# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture
def C():
    @prototype
    class C(NxObject):
        x = mtd.MetaCharacter()
        message = 'Hello, World!'

        @mtd.newtypemethod
        def reset_message(cls):
            cls.message = ''
            return cls

        @mtd.typeinitmethod
        def identity(cls):
            cls.message += 'My x is {0}'.format(cls.x)

        @mtd.typemethod
        def greet(cls):
            print(cls.message)
            return True

        @mtd.metamethod
        def __repr__(cls):
            type_repr = 'C[{}]'.format(cls.x)

            def inst_repr(self):
                return '<' + type_repr + '>'

            return inst_repr

    nxtype(C, x=0)

    return C


def test_registries(C):
    """Tests basic metadescriptor registration mechanics."""
    registries = C[0].metaregistries
    assert C.__metadescriptors__ == registries[mtd.MetaDescriptor]
    assert list(registries[mtd.MetaDescriptor]) == ['x', 'reset_message',
                                                    'identity', 'greet',
                                                    '__repr__']
    assert list(C.__metacharacters__) == ['x']
    assert list(C.__typeinitmethods__) == ['identity']


def test_typemethod(C):
    """Tests method is transferred to the metaclass."""
    inst = C[0]()
    assert C.greet() is True
    assert C[0].greet() is True
    with pytest.raises(AttributeError):
        inst.greet()


def test_new_init_methods(C):
    """Tests the combination of newtype and typeinit methods."""
    assert C[0].message == 'My x is 0'


def test_typeattribute():
    """Tests the default and validate functionalities of TypeAttrbute."""

    @archetype
    class Object(NxObject):
        x = mtd.TypeAttribute(validate=fun.typechecker(int))
        y = mtd.TypeAttribute(default=1)
        z = mtd.TypeAttribute(default=lambda c: c.y)
    Concrete = nxtype(Object, x=0)
    assert Concrete.x == Concrete().x == 0
    assert Concrete.y == Concrete().y == 1
    assert Concrete.z == Concrete().z == 1
    with pytest.raises(mtd.ValidationError):
        nxtype(Object, x=None)


def test_metamethod(C):
    inst = C[0]()
    assert hasattr(type(C), '_set__repr__')
    assert repr(inst) == '<C[0]>'


if __name__ == '__main__':
    pytest.main([__file__])
