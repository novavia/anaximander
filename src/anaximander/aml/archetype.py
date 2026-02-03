# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Declare the archetype decorator for AML prototype classes.

Archetypes define the root behavioral contracts that concrete AML prototypes
inherit. The decorator assigns the archetype role, wires metadata used by the
prototype metaclass, and merges archetype-level declarator types so that
downstream subclasses inherit the correct declarator surface.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from typing import cast

from .prototype import Arche, Archetype, TypeRole, is_archetype, prototype

# endregion

# =============================================================================
# Decorators
# =============================================================================
# region Decorators


def archetype[T: Arche](cls: type[T]) -> type[T]:
    """Declare a prototype class as an AML archetype.

    Only types that directly inherit from an archetype can be declared as archetypes.

    Args:
        cls: The prototype class to declare as an archetype.

    Returns:
        The declared archetype class.

    Raises:
        TypeError: If the class is not a prototype instance.
        TypeError: If the class does not directly inherit from an archetype.
    """
    # Guard the decorator to keep archetype roles well-formed.
    if not isinstance(cls, prototype):
        raise TypeError(f"Expected a prototype instance, got {cls}.")
    if (base := cls.__base__) is None or not is_archetype(base):
        raise TypeError(f"Archetype '{cls.__name__}' must directly inherit from another archetype.")  # noqa
    # Assign archetype role metadata.
    cls.__role__ = TypeRole.ARCHETYPE  # type: ignore[assignment]
    cls.__archetype__ = cast(Archetype, cls)  # type: ignore[assignment]
    # Merge declarator type sets to preserve inheritance semantics.
    base_declarator_types = getattr(base, "__declarator_types__", set())
    if "__declarator_types__" in vars(cls):
        new_declarator_types = getattr(cls, "__declarator_types__")
        setattr(cls, "__declarator_types__", base_declarator_types | set(new_declarator_types))
    else:
        setattr(cls, "__declarator_types__", base_declarator_types)
    return cast(type[T], cls)

# endregion
