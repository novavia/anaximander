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

from anaximander2.utilities.xprops import typedweakproperty

# =============================================================================
# Test Cases
# =============================================================================


def test_weakproperty():

    class C:
        pass

    class D:

        @typedweakproperty(C)
        def obj(self):
            return None

    class X:
        pass

    d = D()
    c = C()
    d.obj = c
    assert d.obj == c
    with pytest.raises(TypeError):
        x = X()
        d.obj = x

if __name__ == '__main__':
    pytest.main([__file__])
