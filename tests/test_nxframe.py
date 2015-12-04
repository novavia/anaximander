#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxregistries.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import unittest
from unittest import TestCase

import nxmeta
import nxframe as nxf

#==============================================================================
### Test Cases
#==============================================================================


class TestTemplate(TestCase):

    class C(nxf.NxFrameworkObject, metaclass=nxf.Template, key='__ctype__',
            _metaclass=nxf.NxFrameworkType):

        def __new__(cls, *args, **kwargs):
            obj = nxf.NxFrameworkObject.__new__(cls, *args, **kwargs)
            obj.created_from_C = True
            return obj

        def __init__(self):
            self.initialized = True

    class D(C):
        i_am_a_d = True

    class E(nxmeta.NxObject, metaclass=nxf.Template, key=('x', 'y')):
        pass

    def setUp(self):
        self.C0 = self.C[0]
        self.D0 = self.D[0]
        self.Exy = self.E['x', 'y']

    def test_create(self):
        assert self.C0.__name__ == 'C_0'
        assert self.D0.__name__ == 'D_0'
        assert self.Exy.__name__ == 'E_x_y'
        assert type(self.C0) == nxf.NxFrameworkType
        assert type(self.D0) == nxf.NxFrameworkType
        assert type(self.Exy) == nxmeta.NxType
        assert self.C0.__ctype__ == 0
        assert self.D0.__ctype__ == 0
        assert self.Exy.x == 'x'
        assert self.Exy.y == 'y'
        assert self.C[0] == self.C0
        assert self.D[0] == self.D0
        assert self.E['x', 'y'] == self.Exy
        assert issubclass(self.D0, self.C0)

    def test_instantiate(self):
        c0 = self.C0()
        d0 = self.D0()
        c1 = self.C(__ctype__=0)
        d1 = self.D(__ctype__=0)
        assert type(c1) == self.C0
        assert type(d1) == self.D0
        assert c0.__ctype__ == 0
        assert d0.__ctype__ == 0
        assert c0.created_from_C
        assert c1.created_from_C
        assert c0.initialized
        assert c1.initialized
        assert d0.created_from_C
        assert d1.created_from_C
        assert d0.initialized
        assert d1.initialized
        assert d0.i_am_a_d
        assert d1.i_am_a_d

if __name__ == '__main__':
    unittest.main()
