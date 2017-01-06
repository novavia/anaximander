#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for schemas.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import marshmallow as msh
import pytest

from anaximander.data import fields, schema as sch

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

# =============================================================================
# Test Cases
# =============================================================================


def test_field_type():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            x = msh.fields.Dict()


def test_restring():
    class MySchema(sch.Schema):
        mac = fields.ReString(pattern=MAC_PATTERN)

    assert MySchema.mac.pattern is MAC_PATTERN
    good_mac = dict(mac='88:4A:EA:69:DF:A2')
    bad_mac = dict(mac='XO:4A:XO:69:DF:XO')
    assert MySchema().validate(good_mac) == {}
    with pytest.raises(sch.ValidationError):
        MySchema().validate(bad_mac)


if __name__ == '__main__':
    pytest.main([__file__])
