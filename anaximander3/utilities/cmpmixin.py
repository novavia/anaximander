#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a Mixin class that provides rich comparison methods.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""


class ComparableMixin:
    """Mixin class to provide comparison methods.

    Usage requires that classes that implement this mixin define a method
    __cmpkey__, which is used as the basis for comparison between instances.
    """

    def _compare(self, other, method):
        try:
            return method(self.__cmpkey__(), other.__cmpkey__())
        except (AttributeError, TypeError):
            msg = "Comparison requires a __cmpkey__ method."
            raise NotImplementedError(msg)

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

    def __hash__(self):
        try:
            return hash(self._cmpkey())
        except (AttributeError, TypeError):
            msg = "ComparableMixin must implement __cmpkey__."
            raise NotImplementedError(msg)
