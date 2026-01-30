"""This module defines base types for AML data and metadata."""

from abc import ABC
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import get_args
from uuid import UUID

type PyScalar = (
    Decimal
    | IntEnum
    | StrEnum
    | UUID
    | bool
    | bytes
    | float
    | int
    | str
)

PY_SCALAR_TYPES = tuple(get_args(PyScalar))

class Metadata(ABC):
    pass

for scalar_type in PY_SCALAR_TYPES:
    try:
        Metadata.register(scalar_type)
    except TypeError:
        pass
