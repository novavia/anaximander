# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Declare the trait decorator for AML prototype classes.

Traits are mixins that contribute declarators and behavior to prototypes while
remaining distinct from archetypes. The trait decorator enforces the role rules
and blocks protodescriptor declarations that would violate trait semantics.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from typing import cast

from .prototype import Arche, TypeRole, is_archetype, is_trait, prototype

# endregion

# =============================================================================
# Decorators
# =============================================================================
# region Decorators


def trait[T: Arche](cls: type[T]) -> type[T]:
    """Declare a prototype class as an AML trait.

    Only types that directly inherit from an archetype or trait can be declared as traits.
    Traits are also not allowed to declare protodescriptors.

    Args:
        cls: The prototype class to declare as a trait.

    Returns:
        The declared trait class.

    Raises:
        TypeError: If the class is not a prototype instance.
        TypeError: If the class does not directly inherit from an archetype or trait.
        TypeError: If the class declares protodescriptors.
    """
    # Guard the decorator to keep trait roles well-formed.
    if not isinstance(cls, prototype):
        raise TypeError(f"Expected a prototype instance, got {cls}.")
    if (base := cls.__base__) is None or not (is_archetype(base) or is_trait(base)):
        raise TypeError(f"Trait '{cls.__name__}' must directly inherit from an archetype or trait.")  # noqa
    # Traits cannot declare protodescriptors.
    declarators = cls.declarators("local")
    if any(declarators[ns] for ns in ("field", "schema", "constructor")):
        raise TypeError("Traits cannot declare or inherit protodescriptors.")
    # Assign trait role metadata.
    cls.__role__ = TypeRole.TRAIT  # type: ignore[assignment]
    return cast(type[T], cls)

# endregion
