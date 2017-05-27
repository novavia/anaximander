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

from collections import OrderedDict

import pytest

import anaximander.utilities.functions as fun
import anaximander.meta.metadescriptors as mtd
from anaximander.meta import nxtype, archetype, prototype, NxObject

# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture
def C():
    @prototype
    class C(NxObject):
        x = mtd.metacharacter()
        message = 'Hello, World!'
        _greetings = 0

        @mtd.newtypemethod
        def reset_message(cls):
            cls.message = ''
            return cls

        @mtd.typeinitmethod
        def identity(cls):
            cls.message += 'My x is {0}'.format(cls.x)
            cls._greetings = 0

        @mtd.typemethod
        def greet(cls):
            print(cls.message)
            cls._greetings += 1
            return True

        @mtd.typeproperty
        def greetings(cls):
            return cls._greetings

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
                                                    'greetings', '__repr__']
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
        x = mtd.TypeAttribute(validate=fun.typecheck(int))
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

    class D(C):
        x = 1

        def __repr__(self):
            return "gotcha"
    
    assert repr(D()) == "gotcha"


def test_typeproperty(C):
    inst = C[0]()
    assert C.greetings == C[0].greetings == inst.greetings == 0
    C[0].greet()
    assert C.greetings == 0
    assert C[0].greetings == inst.greetings == 1


def test_metainstance():

    class C(NxObject):
        c0 = mtd.metainstance(0)
        c1 = mtd.metainstance(0, 1, 2, a=0, b=1, c=2)

        def __init__(self, *args, **kwargs):
            self.arg = args[0]
            self.klen = len(kwargs)

    c0, c1 = C.__metainstances__.values()
    assert all(isinstance(c, C) for c in (c0, c1))
    assert c0.arg == c1.arg == 0
    assert c0.klen == 0
    assert c1.klen == 3

    with pytest.raises(TypeError):
        class C(NxObject):
            c = mtd.metainstance(0)


class MyDescriptor(mtd.TypeDescriptor):
    prop= 'mydescriptors'
    
    def __init__(self, val):
        self.val = val

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        return self.val

def mydescriptor(val):
    return MyDescriptor(val)


def test_typedescriptor():

    class C(NxObject):
        x = mydescriptor(0)

    assert isinstance(C.__mydescriptors__['x'], MyDescriptor)
    c = C()
    assert c.x == 0
    assert c.mydescriptors == OrderedDict([('x', 0)])
    with pytest.raises(AttributeError):
        c.x = 1


if __name__ == '__main__':
    pytest.main([__file__])
