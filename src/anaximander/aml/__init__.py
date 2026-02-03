# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Expose the public Anaximander Modeling Language (AML) API surface.

This package exports the core AML decorators, archetypes, and declarator handles
that define the user-facing DSL. Importing AML also finalizes the base modules
so that declarators and prototype registries are ready for downstream use.

The intent is to provide a concise, stable surface while keeping the underlying
metaclass machinery and registries encapsulated within the package.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports

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

# endregion

# =============================================================================
# Module finalization
# =============================================================================
# region Module finalization

_FINALIZE_MODULE_NAMES: tuple[str, ...] = (
    Object.__module__,
    Data.__module__,
    Model.__module__,
)

for _module_name in _FINALIZE_MODULE_NAMES:
    # Import modules eagerly so AML declarators can finalize their registries.
    try:
        _module = __import__(_module_name, fromlist=["__name__"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import AML module '{_module_name}': {exc}") from exc
    finalize_module(_module)  # type: ignore[arg-type]

# endregion

# =============================================================================
# Public exports
# =============================================================================
# region Public exports

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

# endregion
