"""This module defines the trait decorator."""

from anaximander.aml.meta import Type, TypeRole


def trait(type: Type) -> Type:
    """Declares a type as a trait.

    Only types that directly inherit from an archetype can be declared as traits.

    Args:
        type (Type): The type to declare as a trait.

    Returns:
        Type: The declared trait class.
    """
    if not issubclass(type, Type):
        raise TypeError(f"Expected a Type subclass, got {type.__name__}.")
    base_archetype = type.__bases__[0]
    if base_archetype.__role__ not in (TypeRole.ARCHETYPE, TypeRole.TRAIT):
        raise TypeError(
            f"A type must directly inherit from an archetype or trait to be declared as a trait, "
            f"got {base_archetype.__name__}."
        )
    type.__role__ = TypeRole.TRAIT
    return type
