"""This module sets base classes for admissible data types.

There are two base classes defined here:
- nxdata: this is an abstract base type for data types, starting with all
    the canonical built-in types (i.e. int, str, etc)
- nxmodel: this is an abstract base type for simple mapping of string literals
    to either nxdata or other nxmodel values.
"""

from collections.abc import Mapping
from collections import deque
import dataclasses as dc
import datetime
import decimal
import enum
import ipaddress
import pathlib
from types import GenericAlias
import typing
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    TypedDict,
    _TypedDictMeta as TypedDictMeta,
    Union,
    _Final,
)
import uuid

from beartype._decor.main import beartype
from frozendict import frozendict
from numpy.lib.arraysetops import isin
from pydantic import BaseModel, ValidationError
import numpy as np
import pandas as pd

from ..utilities.functions import singleton
from ..utilities.nxdataclasses import DataClass, pydantic_model_class

from .dataspec import DataSpec


Hint = Union[GenericAlias, _Final]
NoneType = type(None)
HintObject = (GenericAlias, _Final)


class nxdatatype(type):
    """The metaclass for nxdata, which keeps a registry."""

    __types__: ClassVar[set[type]] = {
        NoneType,
        bool,
        int,
        float,
        str,
        bytes,
        datetime.date,
        datetime.time,
        datetime.datetime,
        datetime.timedelta,
        ipaddress.IPv4Address,
        ipaddress.IPv4Interface,
        ipaddress.IPv4Network,
        ipaddress.IPv6Address,
        ipaddress.IPv6Interface,
        ipaddress.IPv6Network,
        enum.Enum,
        enum.IntEnum,
        decimal.Decimal,
        pathlib.Path,
        uuid.UUID,
        list,
        tuple,
        dict,
        set,
        frozenset,
        deque,
        pd.Timestamp,
        pd.Timedelta,
        pd.Period,
    }
    __collections__: ClassVar[set[type]] = {
        list,
        tuple,
        dict,
        set,
        frozenset,
        deque,
    }
    __typing_collections__: ClassVar[dict[type, type]] = {
        typing.List: list,
        typing.Tuple: tuple,
        typing.Dict: dict,
        typing.Set: set,
        typing.FrozenSet: frozenset,
        typing.Deque: deque,
    }
    __collections_typing__: ClassVar[dict[type, type]] = {
        list: typing.List,
        tuple: typing.Tuple,
        dict: typing.Dict,
        set: typing.Set,
        frozenset: typing.FrozenSet,
        deque: typing.Deque,
    }
    __pytypes__: ClassVar[dict[type, type]] = dict()
    __dtypes__: ClassVar[dict[type, str]] = {
        bool: "bool",
        int: "int64",
        float: "float64",
        str: "<U",
        bytes: "S",
        datetime.date: "datetime64[D]",
        datetime.time: None,
        datetime.datetime: "datetime64[ns]",
        datetime.timedelta: "timedelta64",
        ipaddress.IPv4Address: "<U",
        ipaddress.IPv4Interface: "<U",
        ipaddress.IPv4Network: "<U",
        ipaddress.IPv6Address: "<U",
        ipaddress.IPv6Interface: "<U",
        ipaddress.IPv6Network: "<U",
        enum.Enum: None,
        enum.IntEnum: "int64",
        decimal.Decimal: "float64",
        pathlib.Path: "<U",
        uuid.UUID: "<U",
        pd.Timestamp: "datetime64[ns]",
        pd.Timedelta: "timedelta64",
        pd.Period: "period",
    }
    __nptypes__: ClassVar[dict[type, np.dtype]] = {
        bool: np.dtype("bool"),
        int: np.dtype("int64"),
        float: np.dtype("float64"),
        str: np.dtype("<U"),
        bytes: np.dtype("S"),
        datetime.date: np.dtype("datetime64[D]"),
        datetime.time: None,
        datetime.datetime: np.dtype("datetime64[ns]"),
        datetime.timedelta: np.dtype("timedelta64"),
        ipaddress.IPv4Address: np.dtype("<U"),
        ipaddress.IPv4Interface: np.dtype("<U"),
        ipaddress.IPv4Network: np.dtype("<U"),
        ipaddress.IPv6Address: np.dtype("<U"),
        ipaddress.IPv6Interface: np.dtype("<U"),
        ipaddress.IPv6Network: np.dtype("<U"),
        enum.Enum: None,
        enum.IntEnum: np.dtype("int64"),
        decimal.Decimal: np.dtype("float64"),
        pathlib.Path: np.dtype("<U"),
        uuid.UUID: np.dtype("<U"),
        pd.Timestamp: np.dtype("datetime64[ns]"),
        pd.Timedelta: np.dtype("timedelta64"),
        pd.Period: np.dtype("datetime64[ns]"),
    }

    def __instancecheck__(cls, object_: Any) -> bool:
        try:
            type_ = type(object_)
            if cls.from_hint(type_) == type_:
                return True
            return super().__subclasscheck__(cls, type_)
        except TypeError:
            return False

    def __subclasscheck__(cls, type_: type) -> bool:
        try:
            if cls.from_hint(type_) == type_:
                return True
            return super().__subclasscheck__(cls, type_)
        except TypeError:
            return False

    @classmethod
    def from_hint(mcl, hint: Union[type, Hint]) -> type["nxdata"]:
        """This function normalizes and validates hints as nxdata."""
        if isinstance(hint, enum.EnumMeta):
            values = [m._value_ for m in hint.__members__.values()]
            # All values must be of the same field type
            types = set(type(v) for v in values)
            if len(types) == 1:
                type_ = types.pop()
                return mcl.from_hint(type_)
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)
        if origin in mcl.__typing_collections__:
            # This will raise if incorrect specifications are supplied
            for t in args:
                mcl.from_hint(t)
            return mcl.__typing_collections__[origin]
        elif origin in mcl.__collections__:
            # We validate the hint because there is no enforcement
            # when using built-in types
            typing_origin = mcl.__collections_typing__[origin]
            # This will raise TypeError if the hint is wrongly formulated
            typing_origin[args]
            # This will raise if incorrect specifications are supplied
            for t in args:
                mcl.from_hint(t)
            return origin
        elif origin is Union:  # type: ignore
            types = set(args)
            if NoneType in types:
                types -= {NoneType}
                if len(types) == 1:
                    type_ = types.pop()
                    return mcl.from_hint(type_)
        elif origin is Literal:
            # All arguments must be of the same field type
            types = set(type(a) for a in typing.get_args(hint))
            if len(types) == 1:
                type_ = types.pop()
                return mcl.from_hint(type_)
        elif isinstance(hint, type):
            if any(c in mcl.__types__ for c in hint.mro()):
                return hint
            elif issubclass(hint, np.generic):
                return type(hint().item())
        msg = f"Unsupported type hint {hint}"
        raise TypeError(msg)

    @classmethod
    def pytype(mcl, datatype: type["nxdata"]) -> np.dtype:
        """Returns a python object class for a registered data type."""
        origin = typing.get_origin(datatype)
        args = typing.get_args(datatype)
        if origin in mcl.__collections__:
            return origin
        elif origin is Union:  # type: ignore  # origin is 'Optional'
            types = set(args)
            types -= {NoneType}
            return mcl.pytype(types.pop())
        elif origin is Literal:
            return type(args[0])
        elif isinstance(datatype, type):
            try:
                dataspec: DataSpec = getattr(datatype, "dataspec")
            except AttributeError:
                return mcl.__pytypes__.get(datatype, datatype)
            else:
                return dataspec.pytype
        msg = f"Could not determine an object class for {datatype}"
        raise TypeError(msg)

    @classmethod
    def dtype(mcl, datatype: type, **metadata) -> str:
        """Returns a pandas dtype for a registered data type."""
        pytype = mcl.pytype(datatype)
        dtype = mcl.__dtypes__.get(pytype, None)
        if dtype is None:
            msg = f"No dtype matches {datatype}"
            raise TypeError(msg)
        if metadata:
            if dtype.startswith("datetime"):
                if (
                    tz := metadata.get("tz")
                    or metadata.get("tzinfo")
                    or metadata.get("timezone")
                ) is not None:
                    return dtype[:-1] + ", " + str(tz) + "]"
            if dtype == "period":
                if (freq := metadata.get("freq")) is not None:
                    if not isinstance(freq, str):
                        freq = freq.name
                    return f"period[{freq}]"
        return dtype

    @classmethod
    def nptype(mcl, datatype: type) -> np.dtype:
        """Returns the equivalent numpy dtype for a registered data type."""
        pytype = mcl.pytype(datatype)
        dtype = mcl.__nptypes__.get(pytype, None)
        if dtype is None:
            msg = f"No numpy type matches {datatype}"
            raise TypeError(msg)
        return dtype

    @classmethod
    def register(mcl, type_: type, *, pytype=None, nptype=None):
        """Registers a new data type.

        :param type_: The type being registered as data type
        :param pytype: If supplied, the corresponding python type. If not supplied,
            the pytype is equal to type_ itself.
        :param nptype: The corresponding numpy type. If not supplied, it
            defaults to the nptype of the pytype if already registered,
            and if not, conversion to numpy is unsupported.
        """
        mcl.__types__.add(type_)
        if pytype is not None:
            mcl.__pytypes__[type_] = pytype
        if nptype is not None:
            mcl.__nptypes__[type_] = nptype

    def __call__(
        cls,
        data,
        *,
        dataspec: Optional[Hint] = None,
        conform: bool = True,
    ):
        """Redefines call for nxdata, which is abstract and doesn't produce instances."""
        if dataspec is None:
            return data

        @beartype
        def typecheck(data: dataspec):
            return True

        try:
            assert typecheck(data)
        except:
            try:
                assert conform
                nxdata_type = cls.from_hint(dataspec)
                conformed = nxdata_type(data)
                assert typecheck(conformed)
            except:
                msg = f"Incorrect data {data} supplied for {dataspec}"
                raise TypeError(msg)
            else:
                return conformed
        else:
            return data


class nxdata(type, metaclass=nxdatatype):
    """An abstract base class for all data types."""

    pass


def _datatype(cls: type, *, pytype: type = None, nptype: np.dtype = None):
    """Primitive for datatype that registers it as a data type."""
    nxdatatype.register(cls, pytype=pytype, nptype=nptype)
    return cls


def datatype(*, pytype: type = None, nptype: np.dtype = None):
    """A class decorator that registers it as a data type."""

    def decorator(cls: type) -> type:
        return _datatype(cls, pytype=pytype, nptype=nptype)

    return decorator


ModelSpec = Union[
    TypedDictMeta,
    type[DataClass],
    type[BaseModel],
    Mapping[str, type],
]


class nxmodeltype(type):
    """The metaclass for nxmodel."""

    def __instancecheck__(cls, object_: Any) -> bool:
        if issubclass(type(object_), cls):
            return True
        if isinstance(object_, Mapping):
            return all(
                isinstance(k, str) and isinstance(v, (nxdata, cls))
                for k, v in object_.items()
            )

    def __subclasscheck__(cls, type_: type) -> bool:
        try:
            if isinstance(type_, type(cls)):
                return True
            elif hasattr(type_, "__annotations__"):
                if cls.from_spec(type_).__annotations__ == type_.__annotations__:
                    return True
                else:
                    return False
        except TypeError:
            return False

    @classmethod
    def from_spec(mcl, modelspec: ModelSpec) -> TypedDict:
        """This method merely checks the conformity of the specification."""
        try:
            if hasattr(modelspec, "dataschema"):
                dataschema = getattr(modelspec, "dataschema")
                hints = {k: v.hint for k, v in dataschema.fields.items()}
            elif isinstance(modelspec, Mapping):
                hints = modelspec
            elif isinstance(modelspec, TypedDictMeta):
                hints = modelspec.__annotations__
            elif dc.is_dataclass(modelspec):
                hints = {f.name: f.type for f in dc.fields(modelspec)}
            elif issubclass(modelspec, BaseModel):
                hints = {k: v.type_ for k, v in modelspec.__fields__.items()}
            else:
                raise TypeError
        except TypeError:
            msg = f"Cannot create model class from {modelspec}"
            raise TypeError(msg)
        # This raises if the specifications are incorrect
        for h in hints.values():
            datafield_type_from_hint(h)
        return TypedDict(getattr(modelspec, "__name__", "Model"), hints)  # type: ignore

    @classmethod
    def to_pydantic_model_class(mcl, modelspec: ModelSpec) -> type[BaseModel]:
        typed_dict = mcl.from_spec(modelspec)
        hints: dict = typed_dict.__annotations__
        pyhints = {k: pyhint(v) for k, v in hints.items()}
        return pydantic_model_class(modelspec.__name__, annotations=pyhints)

    def __call__(
        cls, data, *, modelspec: Optional[Hint] = None, conform: bool = True, **kwargs
    ):
        """Redefines call for nxmodel, which is abstract and doesn't produce instances."""
        if modelspec is None:
            # In this case we attempt 'schema-on-read'
            if data is not None:
                if isinstance(data, BaseModel):
                    model_class = type(BaseModel)
                elif dc.is_dataclass(modelspec):
                    fields = {f.name: f.type for f in dc.fields(modelspec)}
                    model_class = cls.to_pydantic_model_class(fields)
                elif isinstance(data, Mapping):
                    if kwargs:
                        data = dict(data) | kwargs
                    if isinstance(data, cls):
                        return data
            elif kwargs:
                if isinstance(kwargs, cls):
                    return kwargs
            msg = f"Data supplied to {cls} could not be interpreted as a model"
            raise TypeError(msg)
        else:
            model_class = cls.to_pydantic_model_class(modelspec)
        if data is not None:
            if isinstance(data, BaseModel):
                attrs = data.dict()
            elif dc.is_dataclass(data):
                attrs = dc.asdict(data)
            elif isinstance(data, Mapping):
                attrs = dict(data)
            else:
                msg = f"Could not interpret {data} supplied to {cls}"
                raise TypeError(msg)
        try:
            conformed = model_class(**attrs).dict()
        except ValidationError:
            pass
        else:
            if conform:
                return conformed
            elif attrs == conformed:
                return conformed
        msg = f"Supplied data {attrs} does not conform to specification {modelspec}"
        raise TypeError(msg)


class nxmodel(metaclass=nxmodeltype):
    """Abstract base class for mappings of data and submodels."""

    pass


def _pyhint(hint: Union[type, Hint]) -> type:
    """Primitive for pyhint."""
    if issubclass(hint, nxmodel):
        pyhints = {k: _pyhint(v) for k, v in hint.__annotations__.items()}
        return TypedDict(hint.__name__, pyhints)  # type: ignore
    elif hint in nxdata.__types__:
        return nxdata.__pytypes__.get(hint, hint)
    elif (dataspec := getattr(hint, "dataspec", None)) is not None:
        try:
            return dataspec.pytype
        except AttributeError:
            pass
    elif isinstance(hint, enum.EnumMeta):
        values = [m._value_ for m in hint.__members__.values()]
        # All values must be of the same field type
        types = set(type(v) for v in values)
        if len(types) == 1:
            type_ = types.pop()
            if issubclass(type_, nxdata):
                return hint
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in nxdata.__typing_collections__:
        origin = nxdata.__typing_collections__[origin]
        return origin[tuple(_pyhint(t) for t in args)]
    if origin in nxdata.__collections__:
        # We validate the hint because there is no enforcement
        # when using built-in types
        typing_origin = nxdata.__collections_typing__[origin]
        # This will raise TypeError if the hint is wrongly formulated
        typing_origin[args]
        return origin[tuple(_pyhint(t) for t in args)]
    elif origin is Union:  # type: ignore
        args = set(args)
        if NoneType in args:
            args -= {NoneType}
            if len(args) == 1:
                return Optional[_pyhint(args.pop())]
    elif origin is Literal:
        # All arguments must be of the same field type
        types = set(type(a) for a in typing.get_args(hint))
        if len(types) == 1:
            type_ = types.pop()
            if issubclass(type_, nxdata):
                return _pyhint(hint)
    msg = f"Unsupported type hint {hint}"
    raise TypeError(msg)


def pyhint(hint: Union[type, Hint], *, optional: Optional[bool] = None) -> type:
    """Returns a new hint that replaces typing types with python classes.

    The optional flag controls whether the hint is made optional or not.
    """
    pyhint_ = _pyhint(hint)
    if optional is None:
        return pyhint_
    origin = typing.get_origin(pyhint_)
    if optional is True:
        if origin is Union:  # type: ignore
            return pyhint_
        else:
            return Optional[pyhint_]
    elif optional is False:
        if origin is Union:  # type: ignore
            return typing.get_args(pyhint_)
        else:
            return pyhint_
    raise TypeError


DataFieldType = Union[type[nxdata], type[nxmodel]]


def datafield_type_from_hint(hint: Union[type, Hint, ModelSpec]) -> DataFieldType:
    try:
        if (
            issubclass(hint, (BaseModel, TypedDictMeta))
            or dc.is_dataclass(hint)
            or isinstance(hint, Mapping)
        ):
            return nxmodel.from_spec(hint)
    except TypeError:
        pass
    return nxdata.from_hint(hint)


# Monkey patching of external data types

# pd.Timestamp
def timestamp_parser_validator(cls, value):
    """Pydantic parser-validator for pandas Timestamp class."""
    return pd.to_datetime(value)


def timestamp_get_validators(cls):
    yield timestamp_parser_validator


pd.Timestamp.__get_validators__ = classmethod(timestamp_get_validators)


# pd.Timedelta
def timedelta_parser_validator(cls, value):
    """Pydantic parser-validator for pandas Timedelta class."""
    return pd.to_timedelta(value)


def timedelta_get_validators(cls):
    yield timedelta_parser_validator


pd.Timedelta.__get_validators__ = classmethod(timedelta_get_validators)


# pd.Period
def period_parser_validator(cls, value: Union[tuple, pd.Period]):
    """Pydantic parser-validator for pandas Period class.

    In the general case, the value must be a tuple made of a datetime and frequency.
    """
    if isinstance(value, pd.Period):
        return value
    return pd.Period(*value)


def period_get_validators(cls):
    yield period_parser_validator


pd.Period.__get_validators__ = classmethod(period_get_validators)
