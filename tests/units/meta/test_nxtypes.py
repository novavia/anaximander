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


@nxt.archetype
class Object(BaseObject):
    x = nxd.MetaCharacter()


def test_archetype():
    assert isinstance(Object, nxt.NxType)
    assert isinstance(Object, nxm.ArcheType)
    assert Object.__basename__ == 'Object'
    assert len(Object.nxdescriptors()) == 1
    assert isinstance(Object.nxdescriptors()['x'], nxd.TypeProperty)



#def test_archetype_limitations():
#    """Tests that archetype cannot implement metacharacters."""
#    with pytest.raises(MetaError):
#        @archetype
#        class Object(NxObject):
#            key = mtd.metacharacter()
#
#
#@pytest.fixture
#def arche_clade():
#
#    @archetype
#    class User(NxObject):
#        demographics = mtd.TypeAttribute()
#
#    class UrbanUser(User, demographics='urban'):
#        pass
#
#    return (User, UrbanUser)
#
#
#@pytest.fixture
#def proto_clade():
#
#    @prototype
#    class Object(NxObject):
#        key = mtd.metacharacter()
#        physical = mtd.TypeAttribute(validate=lambda v: isinstance(v, bool))
#
#        @mtd.classtypemethod
#        def __baptize__(mcl, basetype, traits=None, **kwargs):
#            suffix = str(kwargs.get('key', mcl.__type_id__))
#            return mcl.__basename__ + '_' + suffix
#
#
#    class BaseHammer(Object, key='hammer', physical=True):
#        pass
#
#    # Note: this tests conflict between the namespace and kwargs. The
#    # namespace should prevail.
#    class BaseNoise(Object, physical=True):
#        key = 'noise'
#        physical = False
#
#    # This tests that a subtype is properly overriden in Object's cladogram
#    class Hammer(BaseHammer, overtype=True):
#        pass
#
#    class Noise(Object['noise'], overtype=True):
#        pass
#
#    Random = nxtype(Object, key='random')
#
#    return (Object, BaseHammer, BaseNoise, Hammer, Noise, Random)
#
#
#def test_baptism(proto_clade):
#    *_, Random = proto_clade
#    assert Random.__name__ == "Object_random"
#
#
#def test_archetype_inheritance(proto_clade):
#    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
#    assert issubclass(Hammer, Object)
#    assert issubclass(Noise, Object)
#    assert issubclass(Random, Object)
#    assert isinstance(Object, ArcheType)
#    assert isinstance(Object, NxType)
#    assert not isinstance(Hammer, type(Object))
#    assert type(Hammer) is not type(NxObject)
#    assert type(Object).__name__ == 'ObjectProtoType'
#    assert type(Hammer).__name__ == 'ObjectType'
#    assert isinstance(Object, type(Hammer))
#    assert Hammer.__archetype__ is Object
#    assert Object.__archetype__ is Object
#    assert type(Object).__metatype__ == type(Hammer)
#    assert clade(Hammer) is Object
#    assert clade(Noise) is Object
#    assert clade(Random) is Object
#
#
#def test_prototype_instantiation(proto_clade):
#    Object, BaseHammer, _, Hammer, *_ = proto_clade
#    with pytest.raises(TypeError):
#        Object()
#    assert isinstance(Object(key='hammer'), Hammer)
#    assert isinstance(Object(key='hammer'), BaseHammer)
#
#
#def test_typeattribute(proto_clade):
#    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
#    hammer = Hammer()
#    noise = Noise()
#    random = Random()
#    assert 'physical' in type(Object).__typeattributes__
#    assert isinstance(type(Object).physical, property)
#    assert Hammer.physical is True
#    assert Noise.physical is False
#    assert Random.physical is None
#    assert hammer.physical is True
#    assert noise.physical is False
#    assert random.physical is None
#    assert Hammer.typeattributes == ('hammer', True)
#    assert Noise.typeattributes == ('noise', False)
#    with pytest.raises(AttributeError):
#        Hammer.physical = False
#    with pytest.raises(AttributeError):
#        hammer.physical = False
#    with pytest.raises(mtd.ValidationError):
#        nxtype(Object, key='...', physical="don't know")
#
#
#def test_subtype_instantiation(proto_clade):
#    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
#    hammer = Hammer()
#    noise = Noise()
#    random = Random()
#    assert isinstance(hammer, Object)
#    assert isinstance(noise, Object)
#    assert isinstance(random, Object)
#    assert clade(hammer) is Object
#    assert clade(noise) is Object
#    assert clade(random) is Object
#
#
#def test_metacharacters(proto_clade):
#    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
#    assert Object.metacharacters == (None,)
#    assert Hammer.metacharacters == ('hammer',)
#    assert Noise.metacharacters == ('noise',)
#    assert Random.metacharacters == ('random',)
#    # Fail to supply metacharacter 'key'
#    with pytest.raises(MetaError):
#        nxtype(Object)
#
#
#def test_type_registry(arche_clade, proto_clade):
#    User, UrbanUser = arche_clade
#    Object, BaseHammer, BaseNoise, Hammer, Noise, Random = proto_clade
#    with pytest.raises(KeyError):
#        type(User).registry['urban']
#    assert set(User.subtypes) == {UrbanUser}
#    assert Object['hammer'] == Hammer
#    assert set(Object.subtypes) == {Hammer, Noise, Random}



if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
