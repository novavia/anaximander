import datetime
from abc import ABC
from collections.abc import Collection
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel


class Data(ABC):
    pass


class Model(ABC):
    pass


type data = (
    Data
    | bool
    | int
    | float
    | str
    | bytes
    | datetime.date
    | datetime.datetime
    | datetime.time
    | datetime.timedelta
)
# TODO: https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


type model = Model | BaseModel | Dataclass


type dataobject = data | model | Collection[dataobject]
