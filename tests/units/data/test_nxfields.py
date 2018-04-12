#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for schema.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

from anaximander2.data import columns as col
from anaximander2.data import schema as sch
from anaximander2.data.nxfields import NxField, ConformityError

# =============================================================================
# Tests
# =============================================================================


class MyIndex(sch.SchemaIndex):
    id = col.Integer(index='sequential')

    def __rowkey__(self, idx):
        return str(idx[0])

    def __rowidx__(self, key):
        return (int(key),)


def idcol(column):

    class IndexedColumn(sch.IndexedColumn, index=MyIndex, column=column):
        pass

    return IndexedColumn


def test_string_field():
    S = idcol(col.Text())
    f = NxField[S]('hello', id=0)
    assert f.idx == (0,)
    assert f.key == '0'
    assert f.data == 'hello'


def test_category_field():
    S = idcol(col.Categorical(['a', 'b', 'c']))
    f = NxField[S]('a', id=0)
    assert f.idx == (0,)
    assert f.key == '0'
    assert f.data == 'a'
    with pytest.raises(ConformityError):
        NxField[S]('e', id=1)


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
