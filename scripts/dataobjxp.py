from datetime import datetime
from enum import Enum, IntEnum
from multiprocessing.sharedctypes import Value
from typing import ClassVar, Type, TypedDict, Optional
from attr import validate

import pandas as pd
from pydantic import BaseModel
import pytest

from anaximander.utilities.functions import typeproperty
from anaximander import archetype, metaproperty, metadata, metamorph
from anaximander.descriptors.fields import MetadataField, DataField
from anaximander.descriptors.datatypes import nxdata, nxmodel
from anaximander.descriptors import data

from anaximander.dataobjects.base import (
    DataObject,
    Data,
    Integer,
    Float,
    Column,
    Model,
    Entity,
    Table,
    Spec,
    Record,
    Sample,
)


class WeirdInteger(Integer):
    zero: ClassVar[int] = 0

    @data.parser
    def hash_string(cls, value):
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return abs(hash(value))
        raise ValueError

    @data.validator
    def positive(cls, value):
        """Value must be greater than or equal to zero"""
        return value >= cls.zero


class PositiveFloat(Float):
    le = 1e9

    @data.validator
    def strictly_positive(cls, value):
        """Value must be strictly positive"""
        return value > 0


class Point(Spec):
    x: int = data()
    y: int = data()


class XPoint(Spec, strict_schema=False):
    x: int = data()
    y: int = data()


class PositivePoint(Spec):
    x: int = data(gt=0)
    y: int = data(gt=0)


class SimpleFix(Sample):
    mobile: int = data(nx_key=True)
    time: datetime = data(nx_timestamp=True)
    x: float = data()
    y: float = data()


class Fix(Sample):
    mobile: int = data(nx_key=True)
    time: datetime = data(nx_timestamp=True)
    x: float = data()
    y: float = data()
    z: PositiveFloat = data()


class WeirdRecord(Record):
    ten: ClassVar[int] = 10
    key: WeirdInteger = data(nx_key=True)
    i: WeirdInteger = data(ge=10)
    j: WeirdInteger = data()
    inverter: Optional[WeirdInteger] = data()

    @key.validator
    def is_even(cls, value):
        """Value must be even"""
        return value % 2 == 0

    # Here the validators is applied to a WeirdInteger instance
    @key.validator(as_object=True)
    def equal_to_zero_or_greater_than_ten(cls, value):
        return value.data == value.zero or value.data >= cls.ten

    @j.validator
    def greater_than_ten(cls, value):
        """Value must be greater than or equal to 10"""
        return value >= cls.ten

    @inverter.parser
    def invert(cls, value):
        return -value


class DefaultRecord(Record):
    key: WeirdInteger = data(nx_key=True)
    x: PositiveFloat = data(1.0)
    y: PositiveFloat = data(PositiveFloat(1.0))
