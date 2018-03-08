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
        x = nxd.TypeParameter()

    assert isinstance(Object, nxt.NxType)
    assert isinstance(Object, nxm.ArcheType)
    assert Object.__basename__ == 'Object'
    assert len(Object.nxdescriptors()) == 1
    assert isinstance(Object.nxdescriptors()['x'], nxd.TypeAttributeProperty)

    # Archetype subclassing
    @nxt.archetype
    class PhysicalObject(Object):
        y = nxd.TypeParameter()

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
        x: int = nxd.TypeParameter()

        @nxd.typeproperty
        def name(cls):
            return cls.__name__

    with pytest.raises(TypeError):
        Derived.subtype(x='a')

    C = Derived.subtype(x=0)
    assert Derived[0] is C
    assert Derived[0].name == 'Derived'


@nxt.archetype
class MetaMethodObject(BaseObject):
    x: int = nxd.TypeParameter()

    @nxd.newtypemethod
    def modulate_x(cls):
        if cls.x is None or cls.x % 3 == cls.x:
            return cls
        x = cls.x % 3
        return cls.archetype[x]

    @nxd.typeinitmethod
    def square_x(cls):
        if cls.x is None:
            return
        cls.y = cls.x ** 2


def test_metamethods():
    assert 'modulate_x' not in dir(MetaMethodObject)
    C = MetaMethodObject[2]
    assert C.y == 4
    assert 'modulate_x' not in dir(C)
    D = MetaMethodObject[5]
    assert D is C


@nxt.archetype
class A(BaseObject):
    x = nxd.TypeParameter()
    y = nxd.TypeParameter()
    z = nxd.TypeAttribute()


class B(A):
    x = 0


class C(B):
    y = 1


class D(C, overtype=True):
    z = 2


class E(A):
    e = nxd.TypeAttribute()


class F(E):
    e = 3


@nxt.archetype
class G(F):
    x = 0


class H(G):
    y = 1


class I(H, y=2):
    pass


class J(I):
    pass


@nxt.archetype
class K(J):
    x = 2


def test_complex_inheritance():
    assert B.abstract
    assert not C.abstract
    assert C.z is None
    assert not D.abstract
    assert A[0, 1] is D
    assert E.is_pending_archetype
    assert E.abstract
    assert not F.is_pending_archetype
    assert F.abstract
    assert G.is_archetype
    assert G.abstract
    assert not G.is_pending_archetype
    assert G[1] is H
    assert G[2] is I
    assert J.y == 2
    assert K.x == 2
    with pytest.raises(KeyError):
        G.registry[4]
    assert K.registration_key is None
    # This is a violation because we are making a registered subtype
    # of G be the basetype for a new archetype.
    with pytest.raises(nxm.NxMetaError):
        @nxt.archetype
        class L(G, y=3):
            pass


class Param:
    pass


class P(Param):
    pass


class Q(P):
    pass


@nxt.archetype
class Covar(BaseObject):
    x = nxd.TypeParameter(covariant_from=Param)


@nxt.archetype
class ArchParam(BaseObject):
    a = nxd.TypeParameter()


class ArchP(ArchParam):
    a = 0


class ArchQ(ArchP):
    pass


@nxt.archetype
class ArchCovar(BaseObject):
    x = nxd.TypeParameter(covariant_from=ArchParam)


def test_covariance():
    assert Covar[Param] is Covar

    class C(Covar, x=P):
        pass
    assert issubclass(C, Covar[Param])

    class D(Covar, x=Q):
        pass
    assert issubclass(D, C)

    assert ArchCovar[ArchParam] is ArchCovar

    class E(ArchCovar, x=ArchP):
        pass
    assert issubclass(E, ArchCovar)

    class F(E, x=ArchQ):
        pass
    assert issubclass(F, E)

    class G(ArchCovar, x=ArchParam[0], overtype=True):
        pass
    assert issubclass(G, E)
    assert ArchCovar[ArchP] is G

    class H(ArchCovar, x=ArchP):
        pass
    assert issubclass(H, G)
    assert issubclass(H, E)
    assert ArchCovar[ArchP] is G

    class I(ArchCovar, x=ArchParam[1]):
        pass
    assert ArchCovar[ArchParam[1]] is I


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
