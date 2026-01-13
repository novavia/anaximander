"""This module defines the Prototype metaclass for the Anaximander Modeling Language (AML)."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


from enum import Enum
from types import NoneType
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Mapping,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    get_args,
    get_type_hints,
    overload,
)

from annotationlib import Format, get_annotations

from anaximander.utils.funcs import type_name_to_collection_name

from .declarative import AnnotatableDeclarator, DeclarativeNamespace, declarative
from .protodescriptors import (
    MetadescriptorRegistry,
    Protodescriptor,
    ProtodescriptorRegistry,
)

# endregion

# =============================================================================
# Prototype Metaclass
# =============================================================================
# region Prototype Metaclass


class TypeRole(Enum):
    """Enumeration of prototype type roles."""
    ARCHETYPE = "archetype"
    TRAIT = "trait"
    PROTOTYPE = "prototype"


class RegistryState(Enum):
    """Enumeration of registry states that serve as arguments to registry access methods."""
    LOCAL = "local"  # Only locally declared items
    INHERITED = "inherited"  # Only inherited items
    MERGED = "merged"  # Both local and inherited items, merged
    RESOLVED = "resolved"  # Merged items, with defaults applied


class prototypeProtocol(Protocol):
    """Protocol for prototype classes."""
    __archetype__: ClassVar["Archetype"]
    __traits__: ClassVar[tuple["Trait", ...]]
    __local_metadescriptors__: ClassVar[MetadescriptorRegistry]
    __local_metacharacters__: ClassVar[ProtodescriptorRegistry]
    __inherited_metadescriptors__: ClassVar[MetadescriptorRegistry]
    __inherited_metacharacters__: ClassVar[ProtodescriptorRegistry]
    __compilations__: ClassVar[dict[str, dict]]

    @classmethod
    def metadescriptors(cls, *, state: RegistryState = RegistryState.RESOLVED) -> MetadescriptorRegistry:  # noqa
        """The metadescriptors of this type."""
        ...

    @classmethod
    def metacharacters(cls, *, state: RegistryState = RegistryState.RESOLVED) -> ProtodescriptorRegistry:  # noqa
        """The metacharacters of this type."""
        ...


class ArchetypeProtocol(prototypeProtocol):
    """Protocol for archetype classes."""
    __role__: ClassVar[Literal[TypeRole.ARCHETYPE]]

Archetype = type[ArchetypeProtocol]

def is_archetype(cls: type[Any]) -> TypeGuard[Archetype]:
    """Type guard to check if a class is an Archetype."""
    return (isinstance(cls, prototype) and getattr(cls, "__role__", None) == TypeRole.ARCHETYPE)


class TraitProtocol(prototypeProtocol):
    """Protocol for trait classes."""
    __role__: ClassVar[Literal[TypeRole.TRAIT]]
    supertrait: ClassVar["Trait | None"]

Trait = type[TraitProtocol]

def is_trait(cls: type[Any]) -> TypeGuard[Trait]:
    """Type guard to check if a class is a Trait."""
    return (isinstance(cls, prototype) and getattr(cls, "__role__", None) == TypeRole.TRAIT)


class PrototypeProtocol(prototypeProtocol):
    """Protocol for prototype classes."""
    __role__: ClassVar[Literal[TypeRole.PROTOTYPE]]

Prototype = type[PrototypeProtocol]

def is_prototype(cls: type[Any]) -> TypeGuard[Prototype]:
    """Type guard to check if a class is a Prototype."""
    return (isinstance(cls, prototype) and getattr(cls, "__role__", None) == TypeRole.PROTOTYPE)


class Arche(metaclass=declarative):
    """The base class for all AML declartive types."""
    __role__ = TypeRole.ARCHETYPE
    __archetype__: ClassVar[Archetype]
    __traits__ = ()
    _metadescriptors: ClassVar[MetadescriptorRegistry] = MetadescriptorRegistry()
    _metacharacters: ClassVar[ProtodescriptorRegistry] = ProtodescriptorRegistry(_metadescriptors)

    @classmethod
    def metadescriptors(cls, *, state: RegistryState = RegistryState.RESOLVED) -> MetadescriptorRegistry:  # noqa
        """The metadescriptors of this archetype."""
        return cls._metadescriptors

    @classmethod
    def metacharacters(cls, *, state: RegistryState = RegistryState.RESOLVED) -> ProtodescriptorRegistry:  # noqa
        """The metacharacters of this archetype."""
        return cls._metacharacters

Arche.__archetype__ = cast(Archetype, Arche)


class prototype(declarative):
    """Metaclass for declarative types -archetypes, traits and prototypes."""
    __role__: TypeRole
    __archetype__: Archetype  # The prototype's archetype
    __traits__: tuple[Trait, ...]  # The prototype's traits
    __local_metadescriptors__: MetadescriptorRegistry  # The prototype's locally declared metadescriptors  #noqa
    __local_metacharacters__: ProtodescriptorRegistry  # The prototype's locally declared metacharacters  #noqa
    __inherited_metadescriptors__: MetadescriptorRegistry  # The prototype's inherited metadescriptors  #noqa
    __inherited_metacharacters__: ProtodescriptorRegistry  # The prototype's inherited metacharacters  #noqa
    __compilations__: dict[str, dict]  # Holds compilation target handles and parameters

    def __new__(mcls, name, bases, namespace: DeclarativeNamespace, traits=(), **metadata):
        base = bases[0]
        if not issubclass(base, prototype) or base is Arche:
            raise TypeError(f"Base class {base} is not a valid AML prototype base.")
        cls = super().__new__(mcls, name, bases, namespace)
        # Set type annotations on annotatble declarators
        cls.__set_type_annotations__()
        return cls

    def __init__(cls, name, bases, namespace: DeclarativeNamespace, traits=(), **metadata):
        super().__init__(name, bases, namespace, **metadata)
        # Set the base archetype
        base = bases[0]
        base_archetype: Archetype = getattr(base, "__archetype__")
        cls.__archetype__ = base_archetype
        # Next we set and normalize traits
        base_traits: tuple[Trait, ...] = base_archetype.__traits__
        cls.__traits__ = prototype._normalize_traits(*base_traits, *traits)
        # Set metadescriptors
        base_metadescriptors: MetadescriptorRegistry = getattr(base, "__metadescriptors__")
        metadescriptors = MetadescriptorRegistry()


        # TODO: validate metacharacters against archetype definition
        # TODO: validate metacharacter tightening rules

        # Compilations and ast start empty and are filled when the declaring module is evaluated by the AML compiler
        cls.__compilations__: dict[str, dict] = {}
        cls.__ast__ = None
        # Provisionally, new types are assigned the prototype role, but this may be overridden in decorators
        cls.__role__ = TypeRole.PROTOTYPE


    @staticmethod
    def _normalize_traits(*traits: Trait) -> tuple[Trait, ...]:
        """Normalize traits by removing those that are supertraits of others.

        This method also ensures a consistent ordering of traits based on specialization and
        implementation hierarchy.
        The supplied list may contain duplicates and be supplied in an arbitrary order. The
        algorithm visits each trait and uses recursion to ensure that supertraits and implemented
        traits are added to the ordered list before the trait itself.
        """
        # Containers for the ordered traits and visited set
        order = []
        visited = set()

        def visit(T: Trait):
            """Recursively visit traits to build ordered list."""
            if T in visited:
                return
            # First, ensure specialization parent comes first
            if T.supertrait is not None:
                visit(T.supertrait)
            # Then, ensure implemented traits come first, in declared order
            for U in T.__traits__:
                visit(U)
            visited.add(T)
            order.append(T)

        for T in traits:
            visit(T)

        return tuple(order)

    @property
    def archetype(cls) -> Archetype:
        """The archetype of this type."""
        return cls.__archetype__

    @property
    def basetype(cls) -> "prototype":
        """The base class of this type."""
        return cls.__bases__[0]

    @property
    def traits(cls) -> tuple[Trait, ...]:
        """The traits of this type."""
        return tuple(cls.__traits__)

    @property
    def supertrait(cls) -> Trait | None:
        """The supertrait of this trait, if any."""
        if cls.__role__ != TypeRole.TRAIT:
            raise TypeError("Only trait types have a supertrait.")
        base: prototype = cls.__bases__[0]
        if base.__role__ == TypeRole.TRAIT:
            return cast(Trait, base)
        return None

    def metadescriptors(cls, *, state: RegistryState = RegistryState.RESOLVED) -> MetadescriptorRegistry:  # noqa
        """The metadescriptors of this type.

        The method pulls from local and/or inherited metadescriptors based on the specified state.
        """
        return NotImplemented

    def metacharacters(cls, *, state: RegistryState = RegistryState.RESOLVED) -> ProtodescriptorRegistry:  # noqa
        """The metacharacters of this type.

        The method pulls from local and/or inherited metacharacters based on the specified state.
        """
        return NotImplemented

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
        assignable_declarators = [
            declarator for declarator in cls.__declarations__.values()
            if isinstance(declarator, AnnotatableDeclarator)
        ]
        for declarator in assignable_declarators:
            if (name := declarator.name) is None:
                continue
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
            declarator.__set_type__(annotation, hint, nullable)

    @property
    def collection_name(cls):
        """A collection name using camel case and pluralization.

        This can be customized by passing metadata (#TODO).
        """
        return type_name_to_collection_name(cls.__name__)


