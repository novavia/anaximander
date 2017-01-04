#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for functions.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

from anaximander.utilities import nxattr as attr

# =============================================================================
# Test Cases
# =============================================================================


def test_attributes():

    @attr.s
    class C:
        x = attr.ib()
        y = attr.ib()

    # Error raised because of duplication of attribute 'x'
    with pytest.raises(SyntaxError):
        @attr.s
        class D(C):
            x = attr.ib()
            z = attr.ib()

    @attr.s(inherit=False)
    class E(C):
        x = attr.ib()
        z = attr.ib()

    # Note that 'y' is left out (inherit=False), although E does have a
    # class attribute named 'y'.
    assert E.__attrs_attrs__ == (E.x, E.z)


if __name__ == '__main__':
    pytest.main([__file__])
