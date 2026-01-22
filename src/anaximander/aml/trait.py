"""This module defines the trait decorator."""

from typing import cast

from anaximander.aml.prototype import Trait, TypeRole, is_archetype, is_trait, prototype


def trait(cls: prototype) -> Trait:
    """Declares a type as a trait.

    Only types that directly inherit from an archetype or trait can be declared as traits.
    Traits are also not allowed to declare protodescriptors.

    Args:
        cls (prototype): The type to declare as a trait.

    Returns:
        Trait: The declared trait class.
    """
    if not isinstance(cls, prototype):
        raise TypeError(f"Expected a prototype instance, got {cls}.")
    if (base := cls.__base__) is None or not (is_archetype(base) or is_trait(base)):
        raise TypeError(f"Trait '{cls.__name__}' must directly inherit from an archetype or trait.")  # noqa
    # Traits cannot declare protodescriptors
    metacharacters = cls.metacharacters("merged")
    if any(metacharacters[ns] for ns in ("field", "schema", "construction", "data")):
        raise TypeError("Traits cannot declare or inherit protodescriptors.")
    cls.__role__ = TypeRole.TRAIT
    return cast(Trait, cls)
