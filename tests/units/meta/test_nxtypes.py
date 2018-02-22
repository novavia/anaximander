#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxtypes.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

from anaximander2.meta import nxdescriptors as nxd
from anaximander2.meta import nxmetas as nxm
from anaximander2.meta import nxtypes as nxt

# =============================================================================
# Tests type creation
# =============================================================================


class BaseObject(metaclass=nxt.NxType):
    pass


def test_creation():
    assert isinstance(BaseObject, nxt.NxType)

# =============================================================================
# Tests archetype machinery
# =============================================================================


def test_archetypes():

    @nxt.archetype
    class Object(BaseObject):
        x = nxd.MetaCharacter()

    assert isinstance(Object, nxt.NxType)
    assert isinstance(Object, nxm.ArcheType)
    assert Object.__basename__ == 'Object'
    assert len(Object.descriptors()) == 1
    assert isinstance(Object.descriptors()['x'], nxd.ObjectTypeProperty)
    with pytest.raises(TypeError):
        Object()

    # Archetype subclassing
    @nxt.archetype
    class PhysicalObject(Object):
        y = nxd.MetaCharacter()

    assert list(PhysicalObject.__typedescriptors__) == ['x', 'y']
    assert PhysicalObject.x is PhysicalObject.__dict__['_x'] is None
    assert PhysicalObject.y is PhysicalObject.__dict__['_y'] is None

    # Concrete type subclassing
    class Song(Object, x='lala'):
        pass

    assert Song.x == 'lala'
    assert Song().x == 'lala'
    assert isinstance(Object(x='lala'), Song)

    # Subclassing an archetype derivative
    class ReggaeSong(Song, x='tata'):
        pass

    assert issubclass(ReggaeSong, Song)
    assert issubclass(ReggaeSong, Object)
    assert ReggaeSong.x == 'tata'
    assert ReggaeSong().x == 'tata'

    # Overtyping
    class Ragga(ReggaeSong, overtype=True):
        pass

    assert Object['tata'] is Ragga

    # Subclassing complex archetype
    class HammeringObject(PhysicalObject, x='punch'):
        pass

    assert HammeringObject.metacharacters == ('punch', None)
    # No registration with Object since PhysicalObject is an archetype
    with pytest.raises(KeyError):
        Object['punch']

    class Hammer(HammeringObject, y='portable'):
        pass

    assert PhysicalObject['punch', 'portable'] is Hammer
    assert Hammer.metacharacters == ('punch', 'portable')


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
