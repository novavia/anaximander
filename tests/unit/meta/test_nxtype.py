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

from unittest import TestCase

import pytest

from anaximander.registries.folios import Registrable
import anaximander.meta.metadescriptors as mtd
from anaximander.meta.nxmeta import ArcheType, MetaError
from anaximander.meta.nxtype import NxType, nxtype, prototype, clade
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
    key = mtd.MetaCharacter()
    physical = mtd.TypeAttribute(validate=lambda v: isinstance(v, bool))


class BaseHammer(Object, key='hammer', physical=True):
    pass


# Note: this tests conflict between the namespace and kwargs. The
# namespace should prevail.
class BaseNoise(Object, physical=True):
    key = 'noise'
    physical = False


# This tests that a foretype is properly overriden in Object's cladogram
class Hammer(BaseHammer):
    pass


class Noise(Object['noise']):
    pass


Random = nxtype(Object, 'Random', key='random')


def test_archetype_inheritance():
    assert issubclass(Hammer, Object)
    assert issubclass(Noise, Object)
    assert issubclass(Random, Object)
    assert isinstance(Object, ArcheType)
    assert isinstance(Object, NxType)
    assert not isinstance(Hammer, type(Object))
    assert type(Hammer) is not type(NxObject)
    assert type(Object).__name__ == 'ObjectArcheType'
    assert type(Hammer).__name__ == 'ObjectType'
    assert isinstance(Object, type(Hammer))
    assert Hammer.__archetype__ is Object
    assert Object.__archetype__ is Object
    assert type(Object).__metatype__ == type(Hammer)
    assert clade(Hammer) is Object
    assert clade(Noise) is Object
    assert clade(Random) is Object


def test_prototype_instantiation():
    with pytest.raises(TypeError):
        Object()


def test_typeattribute():
    hammer = Hammer()
    noise = Noise()
    random = Random()
    assert 'physical' in type(Object).__typeattributes__
    assert isinstance(type(Object).physical, property)
    assert Hammer.physical is True
    assert Noise.physical is False
    assert Random.physical is None
    assert hammer.physical is True
    assert noise.physical is False
    assert random.physical is None
    assert Hammer.typeattributes == ('hammer', True)
    assert Noise.typeattributes == ('noise', False)
    with pytest.raises(AttributeError):
        Hammer.physical = False
    with pytest.raises(AttributeError):
        hammer.physical = False
    with pytest.raises(mtd.ValidationError):
        nxtype(Object, key='...', physical="don't know")


def test_foretype_instantiation():
    hammer = Hammer()
    noise = Noise()
    random = Random()
    assert isinstance(hammer, Object)
    assert isinstance(noise, Object)
    assert isinstance(random, Object)
    assert clade(hammer) is Object
    assert clade(noise) is Object
    assert clade(random) is Object


def test_metacharacters():
    assert Object.metacharacters == (None,)
    assert Hammer.metacharacters == ('hammer',)
    assert Noise.metacharacters == ('noise',)
    assert Random.metacharacters == ('random',)
    # Fail to supply metacharacter 'key'
    with pytest.raises(MetaError):
        nxtype(Object)


def test_cladogram():
    assert Object['hammer'] == Hammer
    assert set(Object.foretypes) == {Hammer, Noise, Random}


if __name__ == '__main__':
    pytest.main([__file__])
