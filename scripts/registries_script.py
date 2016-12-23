#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for registries.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

from anaximander.registries import folios as fol, registries as nrg

# =============================================================================
# Registrable Item class
# =============================================================================


class Item(fol.RegistrableObject):
    """ A dummy nxobject-like object ."""

    def __init__(self, ix=0):
        self.ix = ix

    def __repr__(self):
        return 'Item[{}]'.format(self.ix)


class SimpleRegistry(nrg.NxRegistry):
    __root__ = fol.NxPortFolio
    __layers__ = [('character', fol.NxPortFolio)]


class ComplexRegistry(nrg.NxRegistry):
    """A registry with mixed layers for testing purposes."""
    __root__ = fol.NxFolder
    __layers__ = [('volume', fol.NxVolume),
                  ('schedule', fol.NxSchedule)]


class RecursiveRegistry(nrg.NxRegistry):
    """A recursive layer registry for testing purposes."""
    __root__ = fol.NxFolder
    __recurse__ = fol.NxFolder


if __name__ == '__main__':
    hierarchy = nrg.Hierarchy('l1', 'l2')
    i0, i1, i2, i3 = (Item(i) for i in range(4))
    hierarchy.register(i0, 'a')
    hierarchy.register(i1, 'a', 'x')
    hierarchy.register(i2, 'b')
    hierarchy.root.title = i3
    assert hierarchy.get(l1='a', l2='x') is i1
    assert hierarchy.fetch('a') == i0
    assert hierarchy['a', 'x'] == i1
