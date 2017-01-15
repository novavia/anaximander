#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a Mixin class that provides rich comparison methods.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""


class ComparableMixin(object):
    """Mixin class to provide comparison methods.

    Usage requires that classes that implement this mixin define a method
    _cmpkey, which is used as the basis for comparison between instances.
    """

    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or return different type.
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)
