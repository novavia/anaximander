"""This module defines the archetype decorator."""

from typing import cast

from anaximander.aml.prototype import Archetype, TypeRole, is_archetype, prototype


def archetype(cls: prototype) -> Archetype:
    """Declares a type as an archetype.

    Only types that directly inherit from an archetype can be declared as archetypes.

    Args:
        cls (prototype): The type to declare as an archetype.

    Returns:
        Archetype: The declared archetype class.
    """
    if not isinstance(cls, prototype):
        raise TypeError(f"Expected a prototype instance, got {cls}.")
    if (base := cls.__base__) is None or not is_archetype(base):
        raise TypeError(f"Archetype '{cls.__name__}' must directly inherit from another archetype.")  # noqa
    cls.__role__ = TypeRole.ARCHETYPE
    cls.__archetype__ = cast(Archetype, cls)
    return cast(Archetype, cls)
