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

import pytest

from anaximander2.utilities import xprops

# =============================================================================
# Test Cases
# =============================================================================


def test_cachedproperty():

    class Type(type):

        @xprops.cachedproperty
        def name(cls):
            return cls.__name__

    class C(metaclass=Type):

        @xprops.cachedproperty
        def prop(self):
            return None

    class D(C):
        pass

    c = C()
    assert not hasattr(C, '_name')
    assert not hasattr(c, '_prop')
    assert C.name == 'C'
    assert c.prop is None
    assert hasattr(C, '_name')
    assert hasattr(c, '_prop')
    assert D.name == 'D'
    assert C.name == 'C'
    del c.prop
    assert not hasattr(c, '_prop')
    xprops.cachedproperty.reset(c)
    # Making sure reset works cleanly on classes where inheritance is involved
    xprops.cachedproperty.reset(D)
    assert '_name' not in D.__dict__
    assert C._name == D._name == 'C'
    xprops.cachedproperty.reset(D)
    assert '_name' not in D.__dict__
    assert C._name == D._name == 'C'
    assert D.name == 'D'


def test_singlesetproperty():

    class C:

        @xprops.singlesetproperty
        def prop(self):
            return None

    c = C()
    assert not hasattr(c, '_prop')
    c.prop = 0
    assert c._prop == 0
    with pytest.raises(AttributeError):
        c.prop = 1
    del c.prop
    c.prop = 1
    assert c._prop == 1


def test_weakproperty():

    class C:
        pass

    class D:

        @xprops.typedweakproperty(C)
        def obj(self):
            return None

    class X:
        pass

    d = D()
    c = C()
    d.obj = c
    assert d.obj == c
    with pytest.raises(TypeError):
        x = X()
        d.obj = x

if __name__ == '__main__':
    pytest.main([__file__])
