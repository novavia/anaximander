"""Helper constructors for AML protodescriptor instances."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any, Literal

from .declarative import MISSING, Missing
from .protodescriptors import BackLinkProtodescriptor, DataProtodescriptor, LinkProtodescriptor

# endregion

# =============================================================================
# Field constructors
# =============================================================================
# region Field constructors


def data(
    default: Any = MISSING,
    *,
    factory: Callable[[], Any] | Missing = MISSING,
    unique: bool = False,
    index: bool = False,
    required: bool = False,
    typekey: bool = False,
    key: bool = False,
    sequence: bool = False,
    timestamp: bool = False,
    start_time: bool = False,
    end_time: bool = False,
    period: bool = False,
    location: bool = False,
    geom: bool = False,
    load: Literal["eager", "lazy"] | Missing = MISSING,
    repr: bool | Callable | str | Missing = MISSING,
    validator: Callable[[Any], bool] | Missing = MISSING,
    gt: Real | Missing = MISSING,
    ge: Real | Missing = MISSING,
    lt: Real | Missing = MISSING,
    le: Real | Missing = MISSING,
    min_length: int | Missing = MISSING,
    max_length: int | Missing = MISSING,
    pattern: str | Missing = MISSING,
    doc: str | Missing = MISSING,
    config: Mapping[str, Any] | None | Missing = MISSING,
) -> DataProtodescriptor:
    """Construct a data protodescriptor with current AML field semantics."""
    return DataProtodescriptor(
        default=default,
        factory=factory,
        validator=validator,
        load=load,
        repr=repr,
        unique=unique,
        index=index,
        required=required,
        typekey=typekey,
        key=key,
        sequence=sequence,
        timestamp=timestamp,
        start_time=start_time,
        end_time=end_time,
        period=period,
        location=location,
        geom=geom,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        doc=doc,
        config=config,
    )


def link(
    *,
    unique: bool = False,
    required: bool = False,
    key: bool = False,
    on_delete: Literal["restrict", "set_null", "cascade"] = "restrict",
    load: Literal["eager", "lazy"] | Missing = MISSING,
    repr: bool | Callable | str | Missing = MISSING,
    validator: Callable[[Any], bool] | Missing = MISSING,
    doc: str | Missing = MISSING,
    config: Mapping[str, Any] | None | Missing = MISSING,
) -> LinkProtodescriptor:
    """Construct a link protodescriptor with current AML field semantics."""
    return LinkProtodescriptor(
        unique=unique,
        key=key,
        on_delete=on_delete,
        required=required,
        load=load,
        repr=repr,
        validator=validator,
        doc=doc,
        config=config,
    )


def backlink(
    *,
    via: type | Missing = MISSING,
    limit: int | Missing = MISSING,
    load: Literal["eager", "lazy"] | Missing = MISSING,
    repr: bool | Callable | str | Missing = MISSING,
    doc: str | Missing = MISSING,
    config: Mapping[str, Any] | None | Missing = MISSING,
) -> BackLinkProtodescriptor:
    """Construct a backlink protodescriptor with current AML field semantics."""
    return BackLinkProtodescriptor(
        via=via,
        limit=limit,
        load=load,
        repr=repr,
        doc=doc,
        config=config,
    )

# endregion
