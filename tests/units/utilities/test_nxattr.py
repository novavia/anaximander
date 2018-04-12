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

from anaximander2.utilities import nxattr as attr
#from anaximander2.meta.nxobject import NxObject
#from anaximander2.meta.nxtype import directory

# =============================================================================
# Test Cases
# =============================================================================


def test_inherit():

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


# XXX: Requires NxObject, which violates dependency hierarchy...
#def test_super():
#
#    i = 0
#
#    @attr.s
#    class C(NxObject, registry=directory('x')):
#        x = attr.ib()
#
#    @attr.s
#    class D(C):
#
#        def __attrs_post_init__(self):
#            nonlocal i
#            i += 1
#
#    c = C(0)
#    d = D(1)
#    assert i == 1
#    assert C[0] == c
#    assert C[1] == D[1] == d


if __name__ == '__main__':
    pytest.main([__file__])
