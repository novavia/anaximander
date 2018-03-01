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
        x = nxd.TypeParameter(key=True)

    assert isinstance(Object, nxt.NxType)
    assert isinstance(Object, nxm.ArcheType)
    assert Object.__basename__ == 'Object'
    assert len(Object.nxdescriptors()) == 1
    assert isinstance(Object.nxdescriptors()['x'], nxd.TypeAttributeProperty)

    # Archetype subclassing
    @nxt.archetype
    class PhysicalObject(Object):
        y = nxd.TypeParameter(key=True)

    assert list(PhysicalObject.__metadescriptors__) == ['x', 'y']
    assert PhysicalObject.x is PhysicalObject.__dict__['_x'] is None
    assert PhysicalObject.y is PhysicalObject.__dict__['_y'] is None

    # Concrete type subclassing
    class Song(Object, x='lala'):
        pass

    assert Song.x == 'lala'
    assert Song().x == 'lala'

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

    assert HammeringObject.typeparameters == ('punch', None)
    # No registration with Object since PhysicalObject is an archetype
    with pytest.raises(KeyError):
        type(Object.archetype).registry['punch']

    class Hammer(HammeringObject, y='portable'):
        pass

    assert PhysicalObject['punch', 'portable'] is Hammer
    assert Hammer.typeparameters == ('punch', 'portable')

    # Type parameter override
    @nxt.archetype
    class Derived(Object):
        x: int = nxd.TypeParameter(key=True)

        @nxd.typeproperty
        def name(cls):
            return cls.__name__

    with pytest.raises(TypeError):
        Derived.subtype(x='a')

    C = Derived.subtype(x=0)
    assert Derived[0] is C
    assert Derived[0].name == 'Derived'


def test_metamethods():

    @nxt.archetype
    class MetaMethodObject(BaseObject):
        x: int = nxd.TypeParameter(key=True)

        @nxd.newtypemethod
        def modulate_x(cls):
            if cls.x is None or cls.x % 3 == cls.x:
                return cls
            x = cls.x % 3
            return cls.archetype[x]

        @nxd.typeinitmethod
        def square_x(cls):
            if cls.x is None:
                pass
            cls.y = cls.x ** 2

    C = MetaMethodObject[2]
    assert C.y == 4
    D = MetaMethodObject[5]
    assert D is C


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
