# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Define AML data archetypes and traits.

Data archetypes represent the elemental scalar types that model fields are built
from. This module defines the core ``Data`` hierarchy, including scalar concrete
types and the ``Measurement`` trait/archetype used to attach physical units.

These archetypes are intentionally small but form the base vocabulary for field
typing and for declarator validation across the AML model layer.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from typing import TYPE_CHECKING, ClassVar

from .archetype import archetype
from .declarators import Metadescriptor, ParserDeclarator, ValidatorDeclarator
from .handles import metadata
from .object import Object
from .trait import trait

# endregion

# =============================================================================
# Data archetypes
# =============================================================================
# region Data archetypes


@archetype
class Data(Object):
    """Represent the AML base archetype for data types."""

    __declarator_types__ = {Metadescriptor, ParserDeclarator, ValidatorDeclarator}


@archetype
class Scalar[T](Data):
    """Represent scalar data types with concrete Python materialization."""
    pass


class Integer(Scalar[int], int):
    """Represent integer scalar data."""
    pass


class Float(Scalar[float], float):
    """Represent float scalar data."""
    pass


class Bool(Scalar[bool]):
    """Represent boolean scalar data."""
    pass


if TYPE_CHECKING:
    # Provide precise typing for checker-only contexts.
    Bool = bool  # type: ignore[assignment]


class String(Scalar[str], str):
    """Represent string scalar data."""
    pass

# endregion


# =============================================================================
# Measurement trait and archetype
# =============================================================================
# region Measurement trait and archetype


@trait
class measurement(Data):
    """Provide a measurement trait that declares a physical unit."""
    unit: ClassVar[str] = metadata()


@archetype
class Measurement(Scalar[float], traits=(measurement,)):
    """Represent measurements with unit metadata and float materialization."""
    pass

# endregion
