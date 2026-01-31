"""This module defines base types for AML data and metadata."""

from abc import ABC
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import Annotated, TypeAliasType, cast, get_args, get_origin
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

def _type_alias_args(tp: object) -> tuple[object, ...]:
    """Return type arguments, unwrapping TypeAliasType when needed."""
    # TypeAliasType stores the real alias value in __value__, so unwrap first.
    if isinstance(tp, TypeAliasType):
        return get_args(tp.__value__)
    return get_args(tp)

PY_SCALAR_TYPES = cast(tuple[type, ...], _type_alias_args(PyScalar))

type PyData = (
    PyScalar
    | tuple["PyData", ...]
    | list["PyData"]
    | dict[str, "PyData"]
    | dict[int, "PyData"]
)


def is_pydata(value: object) -> bool:
    """Return whether a value is a valid PyData instance."""
    # Type objects that are valid PyScalar are accepted as metadata-like values.
    if isinstance(value, type) and value in PY_SCALAR_TYPES:
        return True
    # Scalars are always admissible.
    if isinstance(value, PY_SCALAR_TYPES):
        return True
    # Sequences must be fully PyData.
    if isinstance(value, tuple | list):
        return all(is_pydata(item) for item in value)
    # Dicts must have homogeneous int/str keys and PyData values.
    if isinstance(value, dict):
        key_types = {type(key) for key in value}
        if key_types and key_types not in ({str}, {int}):
            return False
        return all(is_pydata(item) for item in value.values())
    return False


def is_pydata_type(tp: object) -> bool:
    """Return whether a type annotation is a valid PyData type."""
    # Unwrap type aliases to inspect the actual annotation.
    if isinstance(tp, TypeAliasType):
        return is_pydata_type(tp.__value__)
    # Base scalar types are always admissible.
    if isinstance(tp, type) and tp in PY_SCALAR_TYPES:
        return True
    origin = get_origin(tp)
    # Annotated[T, ...] should validate T.
    if origin is Annotated:
        return is_pydata_type(get_args(tp)[0])
    # List[T] must be homogeneous.
    if origin is list:
        args = get_args(tp)
        return len(args) == 1 and is_pydata_type(args[0])
    # Tuple[T, ...] is the homogeneous tuple form we accept.
    if origin is tuple:
        args = get_args(tp)
        return len(args) == 2 and args[1] is Ellipsis and is_pydata_type(args[0])
    # Dict[K, V] requires int/str keys and PyData values.
    if origin is dict:
        args = get_args(tp)
        if len(args) != 2 or args[0] not in (str, int):
            return False
        return is_pydata_type(args[1])
    # Fallback: all union members must be valid.
    if origin is not None:
        return all(is_pydata_type(arg) for arg in get_args(tp))
    return False

def is_metadata_type(tp: object) -> bool:
    """Return whether a type annotation is a valid Metadata type."""
    # Metadata types are either Metadata subclasses or PyData types.
    if isinstance(tp, TypeAliasType):
        return is_metadata_type(tp.__value__)
    if isinstance(tp, type) and issubclass(tp, Metadata):
        return True
    return is_pydata_type(tp)


def is_pydata_container_type(tp: object) -> bool:
    """Return whether a type annotation includes a PyData container."""
    # Accept direct container classes or container generics.
    if isinstance(tp, TypeAliasType):
        return is_pydata_container_type(tp.__value__)
    if isinstance(tp, type) and tp in (list, tuple, dict):
        return True
    origin = get_origin(tp)
    # Annotated[T, ...] should be checked by its base type.
    if origin is Annotated:
        return is_pydata_container_type(get_args(tp)[0])
    # Generic container origins are containers by definition.
    if origin in (list, tuple, dict):
        return True
    # For unions, any container branch qualifies.
    if origin is not None:
        return any(is_pydata_container_type(arg) for arg in get_args(tp))
    return False


def pydata_runtime_type(tp: object) -> type | None:
    """Return the runtime container type for a PyData annotation, if any."""
    # Unwrap type aliases to inspect the actual annotation.
    if isinstance(tp, TypeAliasType):
        return pydata_runtime_type(tp.__value__)
    # Scalars resolve to themselves.
    if isinstance(tp, type) and tp in PY_SCALAR_TYPES:
        return tp
    origin = get_origin(tp)
    # Annotated[T, ...] should resolve by its base type.
    if origin is Annotated:
        return pydata_runtime_type(get_args(tp)[0])
    # Direct container classes count as their runtime type.
    if origin is None and isinstance(tp, type) and tp in (list, tuple, dict):
        return tp
    # Generic container origins resolve to their runtime class.
    if origin in (list, tuple, dict):
        return origin
    # For unions, resolve only if all branches agree on runtime type.
    if origin is not None:
        runtime_types = {pydata_runtime_type(arg) for arg in get_args(tp)}
        runtime_types.discard(None)
        return next(iter(runtime_types)) if len(runtime_types) == 1 else None
    return None


class Metadata(ABC):
    """Base class for metadata and metadata-compatible data."""

    @classmethod
    def __subclasshook__(cls, candidate: type) -> bool:
        if cls is Metadata and (candidate in PY_SCALAR_TYPES or candidate in (tuple, list, dict)):
            return True
        return NotImplemented
