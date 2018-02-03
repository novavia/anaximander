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

from anaximander2.utilities import cmpmixin as cmp

# =============================================================================
# Test Cases
# =============================================================================


class X:

    def __init__(self, x):
        self.x = x


class C(X, cmp.ComparableMixin):

    def __cmpkey__(self):
        return self.x


class Other(X, cmp.ComparableMixin):

    def __cmpkey__(self):
        return self.x


class NoKey(X, cmp.ComparableMixin):
    pass


class NotComparable(X):
    pass


class Derived(C):
    pass


class Liberal(C):
    __cmptypes__ = (object,)


class AttributeComp(X, cmp.ComparableMixin):
    __cmpattrs__ = ('x',)


class LiberalAttributeComp(AttributeComp):
    __cmptypes__ = (object,)


def test_cmpkey():
    c1 = C(0)
    c2 = C(0)
    c3 = C(1)
    assert c1 == c2
    assert c1 != c3
    assert c1 < c3
    assert c3 >= c1


def test_other():
    c1 = C(0)
    l1 = Liberal(0)
    o1 = Other(0)
    assert c1 != o1
    assert l1 == o1


def test_nokey():
    i1 = NoKey(0)
    i2 = NoKey(0)
    assert i1 == i1
    assert i1 != i2
    with pytest.raises(TypeError):
        i1 <= i2


def test_not_comparable():
    i1 = NotComparable(0)
    c1 = C(0)
    d1 = Liberal(0)
    assert i1 == i1
    assert i1 != c1
    assert c1 != i1
    assert d1 != i1


def test_inheritance():
    c1 = C(0)
    d1 = Derived(0)
    assert c1 == d1
    assert d1 == c1


def test_attributes():
    a1 = AttributeComp(0)
    a2 = AttributeComp(1)
    b1 = LiberalAttributeComp(0)
    c1 = C(0)
    assert a1 < a2
    assert a2 >= a1
    assert a1 != c1
    assert b1 == c1


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
