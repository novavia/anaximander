#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxmetas.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import abc
from collections import OrderedDict
import pytest

from anaximander2.utilities import xprops
from anaximander2.meta import NxMetaError
from anaximander2.meta import nxmetas as nxm
from anaximander2.meta import nxdescriptors as nxd

# =============================================================================
# Tests
# =============================================================================


x = nxd.TypeParameter(name='x', key=True, type=int)


class MyType(abc.ABCMeta, metaclass=nxm.NxMeta):
    __archetype__ = None

    def __new__(mcl, name, bases, namespace, **kwargs):
        archetype = mcl.__archetype__
        if archetype is not None:
            base = bases[0]
            # If the base is the archetype, we replace it with __basetype__
            basetype = archetype.__basetype__
            if base is archetype:
                base = basetype
            elif not issubclass(base, basetype):
                msg = "Incorrect use of an ArcheType subclass."
                raise NxMetaError(msg)
            bases = (base,)
        cls = super().__new__(mcl, name, bases, namespace)
        for k, v in mcl.typeattributes.items():
            if k in kwargs:
                setattr(cls, k, kwargs[k])
            else:
                setattr(cls, v.cache, getattr(cls, v.cache, None))
        return cls

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace)
        archetype = cls.__archetype__
        overtype = kwargs.get('overtype', False)
        if archetype is not None:
            if not any(c is None for c in cls.typeparameters):
                overtype = kwargs.get('overtype', False)
                archetype.nxregister(cls, overtype=overtype)

    @xprops.cachedproperty
    def typeparameters(cls):
        """Tuple of type properties for the archetype's parameters."""
        return tuple(getattr(cls, k) for k in type(cls).typeparameters)

    @xprops.cachedproperty
    def registration_key(cls):
        """The key with which cls is registered in its metaclass.

        This can return None (no registration), or a tuple of type parameter
        values that are declared as keys.
        """
        keys = tuple(getattr(cls, k) for k in type(cls).typekeys)
        if any([k is None for k in keys]):
            return None
        elif len(keys) is 0:
            return None
        else:
            return keys


class MyObject(metaclass=MyType):
    pass

MyObject.__metadescriptors__ = OrderedDict([('x', x)])


def test_nxmeta():
    """Tests the nxmeta function."""
    DerivedType = nxm.nxmeta(MyObject, 'Object')
    assert issubclass(DerivedType, MyType)
    assert DerivedType.__basename__ == 'Object'
    assert DerivedType.__name__ == 'ObjectType'
    assert DerivedType.typeparameters == OrderedDict([('x', x)])
    assert isinstance(DerivedType.x, nxd.NxAttribute)


def test_archetype():
    """Tests archetype instantiation."""
    # ArcheType cannot be subclassed
    with pytest.raises(NxMetaError):
        class DerivedArcheType(nxm.ArcheType):
            pass
    MyArchetype = nxm.archmeta(MyObject)
    assert MyArchetype.__basetype__ is MyObject
    assert issubclass(MyArchetype.__metatype__, MyType)
    Object = MyArchetype('Object', (MyObject,), {})
    assert issubclass(Object, MyObject)
    assert isinstance(Object, MyType)
    assert isinstance(Object, MyArchetype)
    assert isinstance(Object.__basetype__.x, nxd.TypeAttributeProperty)
    assert Object.x is None
    with pytest.raises(AttributeError):
        Object.x = 0

    class DerivedObject(Object, x=0):
        pass

    assert isinstance(Object, MyType)
    assert not isinstance(DerivedObject, MyArchetype)
    obj = DerivedObject()
    assert DerivedObject.x == 0
    assert obj.x == 0
    with pytest.raises(AttributeError):
        Object.x = 1
    with pytest.raises(AttributeError):
        obj.x = 1
    assert Object.__archetype__[0] is DerivedObject

    # Testing subclassing of derived type
    class OtherDerivedObject(DerivedObject):
        pass
    # No overtyping, the registry remains unmodified
    assert Object.__archetype__[0] is DerivedObject

    # With overtyping
    class YetOtherDerivedObject(OtherDerivedObject, overtype=True):
        pass
    # With overtyping, the new class takes over the registry
    assert Object.__archetype__[0] is YetOtherDerivedObject

    with pytest.raises(TypeError):
        class IncorrectDerivedObject(Object, x='a'):
            pass

if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
