"""This module defines the trait decorator."""

from typing import cast

from .prototype import Arche, TypeRole, is_archetype, is_trait, prototype


def trait[T: Arche](cls: type[T]) -> type[T]:
    """Declares a type as a trait.

    Only types that directly inherit from an archetype or trait can be declared as traits.
    Traits are also not allowed to declare protodescriptors.

    Args:
        cls (prototype): The type to declare as a trait.

    Returns:
        type[T]: The declared trait class.
    """
    if not isinstance(cls, prototype):
        raise TypeError(f"Expected a prototype instance, got {cls}.")
    if (base := cls.__base__) is None or not (is_archetype(base) or is_trait(base)):
        raise TypeError(f"Trait '{cls.__name__}' must directly inherit from an archetype or trait.")  # noqa
    # Traits cannot declare protodescriptors
    metacharacters = cls.metacharacters("merged")
    if any(metacharacters[ns] for ns in ("field", "schema", "constructor", "data")):
        raise TypeError("Traits cannot declare or inherit protodescriptors.")
    cls.__role__ = TypeRole.TRAIT  # type: ignore[assignment]
    return cast(type[T], cls)
