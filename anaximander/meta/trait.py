"""Traits are mix-in classes for anaximander types.

Traits are written like regular Anaximander types that inherit from an archetype,
and are decorated with the 'trait' decorator. Such a type becomes marked as a
trait, which is an abstract, mix-in class -as such it cannot be instantiated
directly. 
The Anaximander syntactic rules prevent multiple inheritance declarations -a
type only has one base type, either an archetype or a concrete derived type.
However, types are able to inherit from multiple traits. This is enabled by
the use of metacharacters, which specify what traits a type inherits. Metacharacters
are nothing more than a registration string that link an archetype to a trait.
Hence if a type specification contains the metacharacter 'flight', then the
class created from that specification will inherit from a mixin trait that
has been registered with its archetype with the key 'flight'.
The following should be noted:
* Class declarations decorated with @trait are not allowed to set values
for type parameters. Doing so would mess with the resolution of type
specifications, because type parameters can be interpreted to set metacharacters.
Hence allowing metacharacters to in turn set type parameters would create
a circular dependency.
* It is allowable for Traits to declare an __init__ method, though only with
a single 'self' argument. The trait decorator renames the method so that it
doesn't conflict with a type's basic __init__ method, which remains in force.
The Trait's original method is then run after the object has been initialized.
Trait initializations are run in the order that the Traits are declared.
"""

from inspect import signature
from typing import TypeVar

from .nxobject import nxobject
from .nxtype import nxtype, nxtrait, trait_init_name

T = TypeVar("T", bound=nxobject, covariant=True)


def _trait(cls: nxtype[T], metacharacter: str) -> nxtype[T]:
    """The trait decorating function."""
    # Validates basetype
    try:
        assert isinstance(cls, nxtype)
        metadata = cls.typespec.metadata
        assert not metadata
    except AssertionError:
        msg = f"The trait decorator expects an Anaximander type, with unset metadata, not {cls}"
        raise TypeError(msg)

    if "__new__" in vars(cls):
        msg = "Traits cannot redefine __new__"
        raise TypeError(msg)
    # If the decorated class defines an __init__ method, the following steps are
    # applied:
    # * The method must accept a single argument self. This is because it will
    # effectively be run after an object implementing the trait has been initialized,
    # using the __init__ method defined through the primary inheritance chain.
    # Any return value defined by the __init__ method will be ignored at runtime.
    # * The method is renamed so that it does not overload __init__. The new name
    # prefixes the class name in snake case. For instance, for a trait named 'Flight'
    # the method is renamed from __init__ to __flight_init__
    if cls.__original_init__ is not None:
        init_method = cls.__original_init__
        # Enforce signature
        init_signature = signature(init_method)
        if list(init_signature.parameters) != ["self"]:
            msg = f"A trait's init method accepts a single input 'self', not {init_signature}"
            raise TypeError(msg)
        method_name = trait_init_name(cls)
        init_method.__name__ = method_name
        setattr(cls, method_name, init_method)
    delattr(cls, "__init__")

    # Set metaprogramming attributes and associates the trait with its
    # archetype and metacharacter
    nxtrait.register(cls)
    archetype = cls.archetype
    setattr(cls, "__metacharacter__", metacharacter)
    archetype.register_trait(metacharacter, cls)
    return cls


def trait(metacharacter: str):
    """Generates a mix-in class decorator that declares a Trait."""

    def decorator(cls: nxtype[T]) -> nxtype[T]:
        return _trait(cls, metacharacter)

    return decorator
