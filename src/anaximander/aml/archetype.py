"""This module defines the archetype decorator."""

from typing import cast
from anaximander.aml.meta import Type, TypeRole, Archetype


def archetype(type: Type) -> Type:
    """Declares a type as an archetype.

    Only types that directly inherit from an archetype can be declared as archetypes.

    Args:
        type (Type): The type to declare as an archetype.

    Returns:
        Type: The declared archetype class.
    """
    if not issubclass(type, Type):
        raise TypeError(f"Expected a Type subclass, got {type.__name__}.")
    base_archetype = type.__bases__[0]
    traits = type.__bases__[1:]
    if base_archetype.__role__ != TypeRole.ARCHETYPE:
        raise TypeError(
            f"A type must directly inherit from an archetype to be declared as an archetype, "
            f"got {base_archetype.__name__}."
        )
    if traits:
        raise TypeError(
            f"An archetype cannot have traits, got traits {', '.join(t.__name__ for t in traits)}."
        )
    type.__role__ = TypeRole.ARCHETYPE
    type.__archetype__ = cast(Archetype, type)
    return type
