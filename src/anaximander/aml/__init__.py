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
