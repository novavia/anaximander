"""Anaximander AML package initialization."""

from .prototype import prototype, Archetype, Trait, Prototype
from .modules import finalize_module
from .archetype import archetype
from .trait import trait
from .object import Object
from .data import Data, Scalar, Integer, Float, Bool, String, Measurement, measurement
from .model import Model
from .handles import (
    metadata,
    option,
    nxfield,
    meta,
    data,
    link,
    backlink,
    parser,
    validator,
)

_FINALIZE_MODULE_NAMES: tuple[str, ...] = (
    Object.__module__,
    Data.__module__,
    Model.__module__,
)

for _module_name in _FINALIZE_MODULE_NAMES:
    try:
        _module = __import__(_module_name, fromlist=["__name__"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import AML module '{_module_name}': {exc}") from exc
    finalize_module(_module)  # type: ignore[arg-type]


__all__ = [
    "prototype",
    "Archetype",
    "Trait",
    "Prototype",
    "archetype",
    "trait",
    "Object",
    "Data",
    "Scalar",
    "Integer",
    "Float",
    "Bool",
    "String",
    "Measurement",
    "measurement",
    "Model",
    "metadata",
    "option",
    "nxfield",
    "meta",
    "data",
    "link",
    "backlink",
    "parser",
    "validator",
    "finalize_module",
]
