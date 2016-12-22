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

import abc
from unittest import TestCase

import pytest

from anaximander.registries.folios import Registrable
from anaximander.meta.nxmeta import ArcheType
from anaximander.meta.nxtype import NxType, nxtype, prototype
from anaximander.meta.nxobject import NxObject

# =============================================================================
# Test Cases
# =============================================================================


class TestDeclarativeTypeCreation(TestCase):
    """Tests basic declarative metatype / type creation."""

    class ThingType(NxType, basename='Thing'):
        pass

    class Hammer(Registrable, metaclass=ThingType):
        pass

    class RedHammer(Hammer):
        pass

    def test_subclassing(self):
        """Tests subclassing NxType."""
        self.assertTrue(issubclass(self.ThingType, NxType))
        self.assertTrue(issubclass(self.ThingType, Registrable))

    def test_typing(self):
        """Tests creating a new type."""
        self.assertTrue(isinstance(self.Hammer, NxType))


class TestProgrammaticTypeCreation(TestCase):
    """Tests type creation with nxtype."""

    class ThingType(NxType, basename='Thing'):
        pass

    class BaseThing(Registrable, metaclass=ThingType):
        pass

    def test_typing(self):
        new_type = nxtype(self.BaseThing)
        self.assertTrue(issubclass(new_type, self.BaseThing))
        self.assertEqual(new_type.__name__, 'Thing_0')


@prototype
class Object(NxObject):

    @abc.abstractproperty
    def physical(self):
        return NotImplemented


class Hammer(Object):

    @property
    def physical(self):
        return True


class Noise(Object):

    @property
    def physical(self):
        return False


def test_archetype_inheritance():
    assert issubclass(Hammer, Object)
    assert issubclass(Noise, Object)
    assert isinstance(Object, ArcheType)
    assert isinstance(Object, NxType)
    assert not isinstance(Hammer, type(Object))
    assert type(Hammer) is not type(NxObject)
    assert type(Object).__name__ == 'ObjectArcheType'
    assert type(Hammer).__name__ == 'ObjectType'
    assert isinstance(Object, type(Hammer))
    assert Hammer.__archetype__ == Object
    assert Object.__archetype__ == Object


def test_prototype_instantiation():
    with pytest.raises(TypeError):
        Object()


def test_foretype_instantiation():
    hammer = Hammer()
    noise = Noise()
    assert isinstance(hammer, Object)
    assert isinstance(noise, Object)
    assert hammer.physical is True
    assert noise.physical is False


if __name__ == '__main__':
    pytest.main([__file__])
