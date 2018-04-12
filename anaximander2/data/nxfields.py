#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the field archetype, which represents record fields.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from ..meta import archetype, TypeParameter
from .schema import IndexedColumn
from .dataobjects import SingleRowDataObject, fd_type_map, ConformityError


__all__ = []

# =============================================================================
# NxField class
# =============================================================================


@archetype
class NxField(SingleRowDataObject):
    __schema__ = TypeParameter(covariant_from=IndexedColumn)

    @classmethod
    def cast(cls, data):
        dtype = fd_type_map(cls.__schema__.__column__)
        try:
            return dtype(data)
        except ValueError:
            raise ConformityError()

    @property
    def data(self):
        return self._data
