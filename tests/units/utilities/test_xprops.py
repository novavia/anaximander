#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for xprops.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================


import pytest

from anaximander2.meta import NxObject
from anaximander2.utilities.xprops import typedweakproperty

# =============================================================================
# Test Cases
# =============================================================================


def test_weakproperty():

    class C:
    
        @typedweakproperty(NxObject)
        def obj(self):
            return None

    c = C()
    obj = NxObject()
    c.obj = obj
    assert c.obj == obj
    with pytest.raises(TypeError):
        obj = object()
        c.obj = obj

if __name__ == '__main__':
    pytest.main([__file__])
