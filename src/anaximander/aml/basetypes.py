from abc import ABC
from decimal import Decimal
from enum import IntEnum, StrEnum
from uuid import UUID
from typing import  get_args


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

metadata = Metadata | PyScalar
