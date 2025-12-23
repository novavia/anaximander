"""This module defines the Prototype metaclass for the Anaximander Modeling Language (AML)."""

import ast
from abc import ABCMeta
from enum import Enum
from types import NoneType
from typing import (
    ClassVar,
    Literal,
    Mapping,
    Any,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_type_hints,
    overload,
)

from annotationlib import Format, get_annotations

from .protodescriptors import Protodescriptor, AnnotatableDescriptor

from anaximander.utils.funcs import type_name_to_collection_name


P = TypeVar("P", bound=Protodescriptor)


class TypeRole(Enum):
    """Enumeration of prototype type roles."""
    ARCHETYPE = "archetype"
    TRAIT = "trait"
    PROTOTYPE = "prototype"


class ArchetypeProtocol(Protocol):
    """Protocol for archetype classes."""
    __role__: ClassVar[Literal[TypeRole.ARCHETYPE]]

Archetype = type[ArchetypeProtocol]

class TraitProtocol(Protocol):
    """Protocol for trait classes."""
    __role__: ClassVar[Literal[TypeRole.TRAIT]]

Trait = type[TraitProtocol]

class PrototypeProtocol(Protocol):
    """Protocol for prototype classes."""
    __role__: ClassVar[Literal[TypeRole.PROTOTYPE]]

Prototype = type[PrototypeProtocol]


class Arche[T]:
    """The base class for all AML declartive types."""
    __role__: ClassVar[TypeRole] = TypeRole.ARCHETYPE
    __archetype__: ClassVar[Archetype]  # The prototype's archetype

Arche.__archetype__ = cast(Archetype, Arche)


class Type(ABCMeta):
    """Metaclass for declarative types -archetypes, traits and prototypes."""
    __role__: TypeRole
    __archetype__: Archetype  # The prototype's archetype
    __traits__: tuple[Trait, ...]  # The prototype's traits
    __metacharacters__: dict[str, Any]  # The prototype's metacharacters
    __compilations__: dict[str, dict]  # Holds compilation target handles and parameters
    __ast__: ast.ClassDef | None # Holds the prototype's parsed abstract syntax tree

    def __new__(mcls, name, bases, namespace, **metacharacters):
        base = bases[0]
        traits = bases[1:]
        try:
            base_role: TypeRole = getattr(base, "__role__")  # noqa: F841
            archetype: Archetype = getattr(base, "__archetype__")
        except AttributeError:
            raise TypeError(f"Base class {base} is not a valid AML type.")
        # Each trait must be based on the archetype or one of its bases
        archetype_mro = archetype.__mro__
        bad_traits = []
        for trait in traits:
            try:
                assert getattr(trait, "__role__") == TypeRole.TRAIT
                assert getattr(trait, "__archetype__") in archetype_mro
            except AssertionError:
                bad_traits.append(trait.__name__)
        if bad_traits:
            raise TypeError(f"Invalid traits: {bad_traits}")
        cls = super().__new__(mcls, name, bases, namespace)
        return cls

    @staticmethod
    def _normalize_traits(*traits: type) -> tuple[type, ...]:
        normalized = []
        for t in traits:
            if not any(t is not u and issubclass(u, t) for u in traits):
                normalized.append(t)
        return tuple(normalized)
    
    def __init__(cls, name, bases, namespace, **metacharacters):
        super().__init__(name, bases, namespace)
        base = bases[0]
        traits = bases[1:]
        base_archetype: Archetype = getattr(base, "__archetype__")
        base_traits: tuple[Trait, ...] = getattr(base, "__traits__", tuple())
        base_metacharacters: dict[str, Any] = getattr(base, "__metacharacters__", {})
        # Set archetype
        cls.__archetype__ = base_archetype
        # Set traits
        cls.__traits__ = Type._normalize_traits(*traits, *base_traits)
        # Set metacharacters
        # TODO: validate metacharacters against archetype definition
        # TODO: validate metacharacter tightening rules
        cls.__metacharacters__ = {**base_metacharacters, **metacharacters}
        # Compilations and ast start empty and are filled when the declaring module is evaluated by the AML compiler
        cls.__compilations__: dict[str, dict] = {}
        cls.__ast__ = None
        # Provisionally, new types are assigned the prototype role, but this may be overridden in decorators
        cls.__role__ = TypeRole.PROTOTYPE


    @overload
    def protodescriptors(cls, *, inherited: bool = True) -> Mapping[str, Protodescriptor]: ...
    @overload
    def protodescriptors(cls, *types: type[P], inherited: bool = True) -> Mapping[str, P]: ...

    def protodescriptors(
        cls,
        *types: type[Protodescriptor],
        inherited: bool = True,
    ) -> Mapping[str, Protodescriptor]:
        """Returns a dictionary of protodescriptors of the supplied types.

        If inherited is set to True, protodescriptors declared in parent models are included.
        Otherwise, only the protodescriptors directly declared by cls are returned.
        """
        selected_types: tuple[type[Protodescriptor], ...] = types or (Protodescriptor,)
        cls_protodescriptors = {
            k: v for k, v in cls.__dict__.items() if isinstance(v, selected_types)
        }
        if inherited:
            parent = cls.mro()[1]
            if isinstance(parent, Type):
                parent_protodescriptors = dict(parent.protodescriptors(*selected_types, inherited=True))
                protodescriptors = parent_protodescriptors | cls_protodescriptors
            else:
                protodescriptors = cls_protodescriptors
        else:
            protodescriptors = cls_protodescriptors
        return protodescriptors

    @staticmethod
    def __unwrap_optional_type__(type_hint: Any) -> tuple[Any, bool]:
        """Detect optional annotations and return base type with a nullable flag."""
        if type_hint is None:
            return None, False
        if type_hint is NoneType:
            return None, True
        args = get_args(type_hint)
        if args and any(arg is NoneType for arg in args):
            non_none_args = [arg for arg in args if arg is not NoneType]
            base_type = non_none_args[0] if non_none_args else None
            return base_type, True
        return type_hint, False

    def __set_type_annotations__(cls):
        """Sets type annotations on typed protodescriptors."""
        annotation_values = get_annotations(cls, format=Format.VALUE)
        annotation_strings = get_annotations(cls, format=Format.STRING)
        superhints = get_type_hints(cls)  # This includes super classes
        protodescriptors = cls.protodescriptors(AnnotatableDescriptor, inherited=False)
        for name, protodescriptor in protodescriptors.items():
            annotation_value = annotation_values.get(name, "")
            annotation_string = annotation_strings.get(name, "")
            if isinstance(annotation_value, str):
                annotation = f'"{annotation_value}"'
            else:
                annotation = annotation_string
            # TODO: normalize to a prototype in case of model attributes
            # TODO: normalize to a metadata type in case of option / metacharacter
            hint = superhints.get(name, None)
            hint, nullable = cls.__unwrap_optional_type__(hint)
            protodescriptor.__set_type__(annotation, hint, nullable)

    @property
    def collection_name(cls):
        """A collection name using camel case and pluralization.

        This can be customized by passing metadata (#TODO).
        """
        return type_name_to_collection_name(cls.__name__)
    

