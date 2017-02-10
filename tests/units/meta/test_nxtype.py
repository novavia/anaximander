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

from anaximander.utilities import nxattr
from anaximander.registries.folios import Registrable
import anaximander.meta.metadescriptors as mtd
from anaximander.meta.nxmeta import ArcheType, MetaError
from anaximander.meta.nxtype import NxType, nxtype, archetype, prototype, \
                                    clade, NoRegistry, Directory, directory
from anaximander.meta.nxobject import NxObject

# =============================================================================
# Tests type creation
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

# =============================================================================
# Tests archetype / prototype machinery
# =============================================================================


def test_archetype_limitations():
    """Tests that archetype cannot implement metacharacters."""
    with pytest.raises(MetaError):
        @archetype
        class Object(NxObject):
            key = mtd.MetaCharacter()


@pytest.fixture
def arche_clade():

    @archetype
    class User(NxObject):
        demographics = mtd.TypeAttribute()

    class UrbanUser(User, demographics='urban'):
        pass

    return (User, UrbanUser)


@pytest.fixture
def proto_clade():

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

    # This tests that a subtype is properly overriden in Object's cladogram
    class Hammer(BaseHammer):
        pass

    class Noise(Object['noise']):
        pass

    Random = nxtype(Object, name='Random', key='random')

    return (Object, BaseHammer, BaseNoise, Hammer, Noise, Random)


def test_archetype_inheritance(proto_clade):
    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
    assert issubclass(Hammer, Object)
    assert issubclass(Noise, Object)
    assert issubclass(Random, Object)
    assert isinstance(Object, ArcheType)
    assert isinstance(Object, NxType)
    assert not isinstance(Hammer, type(Object))
    assert type(Hammer) is not type(NxObject)
    assert type(Object).__name__ == 'ObjectProtoType'
    assert type(Hammer).__name__ == 'ObjectType'
    assert isinstance(Object, type(Hammer))
    assert Hammer.__archetype__ is Object
    assert Object.__archetype__ is Object
    assert type(Object).__metatype__ == type(Hammer)
    assert clade(Hammer) is Object
    assert clade(Noise) is Object
    assert clade(Random) is Object


def test_prototype_instantiation(proto_clade):
    Object, *_ = proto_clade
    with pytest.raises(TypeError):
        Object()


def test_typeattribute(proto_clade):
    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
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


def test_subtype_instantiation(proto_clade):
    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
    hammer = Hammer()
    noise = Noise()
    random = Random()
    assert isinstance(hammer, Object)
    assert isinstance(noise, Object)
    assert isinstance(random, Object)
    assert clade(hammer) is Object
    assert clade(noise) is Object
    assert clade(random) is Object


def test_metacharacters(proto_clade):
    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
    assert Object.metacharacters == (None,)
    assert Hammer.metacharacters == ('hammer',)
    assert Noise.metacharacters == ('noise',)
    assert Random.metacharacters == ('random',)
    # Fail to supply metacharacter 'key'
    with pytest.raises(MetaError):
        nxtype(Object)


def test_type_registry(arche_clade, proto_clade):
    User, UrbanUser = arche_clade
    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
    with pytest.raises(KeyError):
        type(User).registry['urban']
    assert set(User.subtypes) == {UrbanUser}
    assert Object['hammer'] == Hammer
    assert set(Object.subtypes) == {Hammer, Noise, Random}

# =============================================================================
# Tests instance registration
# =============================================================================


def test_no_registry():
    Object = nxtype(NxObject)
    assert Object.__registry__ is NxObject.__registry__
    assert isinstance(Object.__registry__, NoRegistry)


def test_directory():

    class Object(NxObject, registry=directory('x')):
        def __init__(self, x):
            self.x = x
            super().__init__()

    assert isinstance(Object.__registry__, Directory)
    obj = Object('a')
    assert Object['a'] is obj

    class C1(Object):
        pass

    class C2(Object, registry=directory('x')):
        pass

    c1 = C1('a')
    c2 = C2('a')

    assert Object['a'] is C1['a'] is c1
    assert C2['a'] is c2

if __name__ == '__main__':
    pytest.main([__file__])
