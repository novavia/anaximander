"""This module defines the archetype decorator."""

from typing import cast
from anaximander.aml.prototype import prototype, TypeRole, Archetype


def archetype(type_: prototype) -> prototype:
    """Declares a type as an archetype.

    Only types that directly inherit from an archetype can be declared as archetypes.

    Args:
        type (Type): The type to declare as an archetype.

    Returns:
        Type: The declared archetype class.
    """
    if not isinstance(type_, prototype):
        raise TypeError(f"Expected a Type instance, got {type_.__name__}.")
    base_archetype = type_.basetype
    traits = type_.traits
    if base_archetype.__role__ != TypeRole.ARCHETYPE:
        raise TypeError(
            f"A type must directly inherit from an archetype to be declared as an archetype, "
            f"got {base_archetype.__name__}."
        )
    if traits:
        raise TypeError(
            f"An archetype cannot have traits, got traits {', '.join(t.__name__ for t in traits)}."
        )
    type_.__role__ = TypeRole.ARCHETYPE
    type_.__archetype__ = cast(Archetype, type_)
    return type_
