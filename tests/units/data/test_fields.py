#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for fields.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import datetime as dt

import marshmallow as msh
import pandas as pd
import pytest

from anaximander.utilities.nxtime import datetime, UTC
from anaximander.utilities import nxrange as rge
from anaximander.data import fields, schema as sch, quantities as qnt, NxScalar

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
    assert MySchema.mac.interval == rge.string_interval


def test_scalar():
    speed = qnt.Quantity('speed', 'mph')

    class SpeedMPH(NxScalar):
        quantity = speed

    class MySchema(sch.Schema):
        mph = fields.Scalar(SpeedMPH)

    data = {'mph': 65.}
    obj = MySchema().load(data).data
    assert str(obj['mph']) == '65.0 mph'
    assert MySchema.mph.pygetattr(obj) == ('mph', 65.)
    assert MySchema.mph.ypgetattr({'mph': 65.}) == ('mph', SpeedMPH(65.))
    dump = MySchema().dump(obj).data
    assert dump == data
    assert type(dump['mph']) is float
    assert MySchema.mph.interval == rge.float_interval

    bad_data = {'mph': '?'}
    with pytest.raises(sch.ValidationError):
        MySchema().load(bad_data)


def test_timestamp():

    class MySchema(sch.Schema):
        timestamp = fields.Timestamp()

    timestring = '2017-02-17 01:55:27.985550+00:00'
    pydatetime = dt.datetime(2017, 2, 17, 1, 55, 27, 985550, tzinfo=UTC)
    data = {'timestamp': timestring}
    obj = MySchema().load(data).data
    assert obj['timestamp'] == datetime(timestring)
    assert MySchema.timestamp.pygetattr(obj) == ('timestamp', pydatetime)
    dump = MySchema().dump(obj).data
    assert dump == data


def test_duration():

    class MySchema(sch.Schema):
        duration = fields.Duration()

    dstring = '5 minutes'
    timedelta = dt.timedelta(minutes=5)
    data = {'duration': dstring}
    obj = MySchema().load(data).data
    assert obj['duration'] == pd.Timedelta(dstring)
    assert MySchema.duration.pygetattr(obj) == ('duration', timedelta)
    dump = MySchema().dump(obj).data
    assert dump['duration'] == '0 days 00:05:00'


def test_period():

    class MySchema(sch.Schema):
        period = fields.Period('5min')

    pstring = '2017-02-17 02:00'
    pydatetime = dt.datetime(2017, 2, 17, 2, 0)
    data = {'period': pstring}
    obj = MySchema().load(data).data
    assert obj['period'] == pd.Period(pstring, freq='5min')
    assert MySchema.period.pygetattr(obj) == ('period', pydatetime)
    dump = MySchema().dump(obj).data
    assert dump['period'] == '2017-02-17 02:00'

if __name__ == '__main__':
    pytest.main([__file__])
