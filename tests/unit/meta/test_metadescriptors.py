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

import anaximander.meta.metadescriptors as mtd
from anaximander.meta.nxtype import nxtype, prototype
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

        @mtd.metamethod
        def greet(cls):
            print(cls.message)
            return True

    nxtype(C, x=0)

    return C


def test_registries(C):
    """Tests basic metadescriptor registration mechanics."""
    registries = C[0].metaregistries
    assert C.__metadescriptors__ == registries[mtd.MetaDescriptor]
    assert list(registries[mtd.MetaDescriptor]) == ['x', 'reset_message',
                                                    'identity', 'greet']
    assert list(C.__metacharacters__) == ['x']
    assert list(C.__typeinitmethods__) == ['identity']


def test_metamethod(C):
    """Tests method is transferred to the metaclass."""
    inst = C[0]()
    assert C.greet() is True
    assert C[0].greet() is True
    with pytest.raises(AttributeError):
        inst.greet()


def test_new_init_methods(C):
    """Tests the combination of newtype and typeinit methods."""
    assert C[0].message == 'My x is 0'


if __name__ == '__main__':
    pytest.main([__file__])
