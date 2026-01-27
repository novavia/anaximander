"""This module defines the archetype decorator."""

from typing import cast

from .prototype import Arche, Archetype, TypeRole, is_archetype, prototype


def archetype[T: Arche](cls: type[T]) -> type[T]:
    """Declares a type as an archetype.

    Only types that directly inherit from an archetype can be declared as archetypes.

    Args:
        cls (type[T]): The type to declare as an archetype.

    Returns:
        type[T]: The declared archetype class.
    """
    if not isinstance(cls, prototype):
        raise TypeError(f"Expected a prototype instance, got {cls}.")
    if (base := cls.__base__) is None or not is_archetype(base):
        raise TypeError(f"Archetype '{cls.__name__}' must directly inherit from another archetype.")  # noqa
    cls.__role__ = TypeRole.ARCHETYPE  # type: ignore[assignment]
    cls.__archetype__ = cast(Archetype, cls)  # type: ignore[assignment]
    # Next we merge the declarator types from the base archetype
    base_declarator_types = getattr(base, "__declarator_types__", set())
    if "__declarator_types__" in vars(cls):
        new_declarator_types = getattr(cls, "__declarator_types__")
        setattr(cls, "__declarator_types__", base_declarator_types | set(new_declarator_types))
    else:
        setattr(cls, "__declarator_types__", base_declarator_types)
    return cast(type[T], cls)
