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
import pytest

from anaximander2.meta import NxMetaError
from anaximander2.meta import nxmetas as nxm

# =============================================================================
# Tests
# =============================================================================


class MyType(abc.ABCMeta):
    pass


class MyObject(metaclass=MyType):
    pass


def test_nxmeta():
    """Tests the nxmeta function."""
    DerivedType = nxm.nxmeta(MyObject, 'Object')
    assert issubclass(DerivedType, MyType)
    assert DerivedType.__basename__ == 'Object'
    assert DerivedType.__name__ == 'ObjectType'


def test_archetype():
    """Tests archetype instantiation."""
    # ArcheType cannot be subclassed
    with pytest.raises(NxMetaError):
        class DerivedArcheType(nxm.ArcheType):
            pass
    MyArchetype = nxm.archmeta(MyObject)
    assert MyArchetype.__basetype__ is MyObject
    assert issubclass(MyArchetype.__metatype__, MyType)


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
