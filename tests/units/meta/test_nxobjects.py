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
from anaximander2.meta import nxtypes as nxt
from anaximander2.meta.nxobjects import NxObject


# =============================================================================
# Tests
# =============================================================================


class C(NxObject):
    pass


def test_basic_object():
    assert isinstance(C, nxt.NxType)


@nxt.archetype
class Object(NxObject):
    x: int = nxd.TypeParameter(key=True)
    y: str = nxd.TypeParameter(key=True)


def test_new():
    obj = Object(x=0, y='a')
    assert isinstance(obj, Object)
    assert isinstance(obj, Object[0, 'a'])


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
