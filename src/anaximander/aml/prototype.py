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
    Literal,
    Protocol,
    TypeGuard,
    cast,
    get_args,
    get_type_hints,
)

from annotationlib import Format, get_annotations

from anaximander.utils.funcs import type_name_to_collection_name

from .declarative import AnnotatableDeclarator, DeclarativeNamespace, declarative
from .metadescriptors import Metadescriptor, MetadescriptorRegistry
from .protodescriptors import ProtodescriptorRegistry

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


class prototypeProtocol(Protocol):
    """Protocol for prototype classes."""
    __archetype__: ClassVar["Archetype"]
    __traits__: ClassVar[tuple["Trait", ...]]
    __merged_traits__: ClassVar[tuple["Trait", ...]]
    __metadescriptors__: ClassVar[MetadescriptorRegistry]
    __metacharacters__: ClassVar[ProtodescriptorRegistry]
    __merged_metadescriptors__: ClassVar[MetadescriptorRegistry]
    __merged_metacharacters__: ClassVar[ProtodescriptorRegistry]
    __compilations__: ClassVar[dict[str, dict]]

    @classmethod
    def implements(cls, metatype: "Archetype | Trait") -> bool:
        """Checks if this type implements the specified archetype or trait."""
        ...

    @classmethod
    def metadescriptors(cls, view: Literal["local", "merged", "resolved"]) -> MetadescriptorRegistry:  # noqa
        """The metadescriptors of this type."""
        ...

    @classmethod
    def metacharacters(cls, view: Literal["local", "merged", "resolved"]) -> ProtodescriptorRegistry:  # noqa
        """The metacharacters of this type."""
        ...


class ArchetypeProtocol(prototypeProtocol):
    """Protocol for archetype classes."""
    __role__: ClassVar[Literal[TypeRole.ARCHETYPE]]
    __declarator_types__: ClassVar[set[type[Metadescriptor]]]

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

def conforms(trait: Trait, archetype: Archetype) -> bool:
    """Checks if a trait conforms to the specified archetype."""
    if not is_trait(trait):
        raise TypeError(f"Expected a Trait type, got {trait}.")
    if not is_archetype(archetype):
        raise TypeError(f"Expected an Archetype type, got {archetype}.")
    # The trait conforms if its archetype is a parent of the specified archetype
    trait_archetype = trait.__archetype__
    return trait_archetype in archetype.mro()


class PrototypeProtocol(prototypeProtocol):
    """Protocol for prototype classes."""
    __role__: ClassVar[Literal[TypeRole.PROTOTYPE]]

Prototype = type[PrototypeProtocol]

def is_prototype(cls: type[Any]) -> TypeGuard[Prototype]:
    """Type guard to check if a class is a Prototype."""
    return (isinstance(cls, prototype) and getattr(cls, "__role__", None) == TypeRole.PROTOTYPE)


class Arche(metaclass=declarative):
    """The base class for all AML declartive types."""
    __role__: ClassVar[Literal[TypeRole.ARCHETYPE]] = TypeRole.ARCHETYPE
    __archetype__: ClassVar[Archetype]
    __traits__: ClassVar[tuple[Trait, ...]] = ()
    _metadescriptors: ClassVar[MetadescriptorRegistry] = MetadescriptorRegistry()
    _metacharacters: ClassVar[ProtodescriptorRegistry] = ProtodescriptorRegistry(_metadescriptors)
    __declarator_types__: ClassVar[set[type[Metadescriptor]]] = {Metadescriptor}

    def __new__(cls, *args, **kwargs):
        raise TypeError("Archetypes, traits and prototypes cannot be instantiated directly.")

    @classmethod
    def traits(cls, view: Literal["local", "merged", "total"]) -> tuple[Trait, ...]:
        """The traits of this archetype."""
        return ()

    @classmethod
    def metadescriptors(cls, view: Literal["local", "merged", "resolved"]) -> MetadescriptorRegistry:  # noqa
        """The metadescriptors of this archetype."""
        return cls._metadescriptors.copy()

    @classmethod
    def metacharacters(cls, view: Literal["local", "merged", "resolved"]) -> ProtodescriptorRegistry:  # noqa
        """The metacharacters of this archetype."""
        return cls._metacharacters.copy()

Arche.__archetype__ = cast(Archetype, Arche)


class prototype(declarative):
    """Metaclass for declarative types -archetypes, traits and prototypes."""
    __role__: TypeRole
    __archetype__: Archetype  # The prototype's archetype
    __traits__: tuple[Trait, ...]  # The prototype's traits implemented on top of its archetype
    __merged_traits__: tuple[Trait, ...]  # The prototype's merged traits, including inherited ones
    __metadescriptors__: MetadescriptorRegistry  # The prototype's locally declared metadescriptors
    __metacharacters__: ProtodescriptorRegistry  # The prototype's locally declared metacharacters
    __merged_metadescriptors__: MetadescriptorRegistry  # The prototype's merged metadescriptors
    __merged_metacharacters__: ProtodescriptorRegistry  # The prototype's merged metacharacters
    __compilations__: dict[str, dict]  # Holds compilation target handles and parameters

    def __new__(mcls, name, bases, namespace: DeclarativeNamespace, traits=(), **metadata):
        base = bases[0]
        if not (isinstance(base, prototype) or base is Arche):
            raise TypeError(f"Base class {base} is not a valid AML prototype base.")
        if any(isinstance(base, prototype) for base in bases[1:]):
            raise TypeError("Prototypes do not support multiple inheritance.")
        cls = super().__new__(mcls, name, bases, namespace)
        # Set type annotations on annotatble declarators
        cls.__set_type_annotations__()

        return cls

    def __init__(cls, name, bases, namespace: DeclarativeNamespace, traits=(), **metadata):
        super().__init__(name, bases, namespace, **metadata)
        # Set the base archetype
        base: prototype | type[Arche]= bases[0]
        base_archetype: Archetype = getattr(base, "__archetype__")
        archetype = cls.__archetype__ = base_archetype
        # Next we normalize and set traits
        # Merged traits include those of the base type
        cls.__merged_traits__ = prototype._normalize_traits(cls, *traits)
        # Local traits are those that concretely specialize the base
        base_traits = base.traits("merged")
        cls.__traits__ = tuple(T for T in cls.__merged_traits__ if T not in base_traits)
        # Validate that all declarations conform to the archetype
        allowed_declarator_types = tuple(getattr(archetype, "__declarator_types__", set()))
        for declarator in cls.__declarations__.values():
            if not isinstance(declarator, allowed_declarator_types):
                dtype = type(declarator).__handle__ or type(declarator).__name__
                msg = f"Declarator of type '{dtype}' is not allowed in archetype {archetype.__name__}."  # noqa
                raise TypeError(msg)
        # Set local metadescriptors
        cls.__metadescriptors__ = MetadescriptorRegistry()
        for declarator in cls.__declarations__.values():
            if isinstance(declarator, Metadescriptor):
                cls.__metadescriptors__.register(declarator.name, declarator)
        # Merge inherited metadescriptors
        merged_metadescriptors = base.metadescriptors("merged")
        for trait in cls.__traits__:
            trait_metadescriptors = trait.metadescriptors("merged")
            merged_metadescriptors.update(trait_metadescriptors)
        merged_metadescriptors.update(cls.__metadescriptors__)
        cls.__merged_metadescriptors__ = merged_metadescriptors
        # Set local metacharacters
        cls.__metacharacters__ = ProtodescriptorRegistry(cls.__merged_metadescriptors__)
        for declarator in cls.__declarations__.values():
            if not isinstance(declarator, Metadescriptor):
                cls.__metacharacters__.register(declarator.name, declarator)
        for name, value in cls.__bindings__.items():
            cls.__metacharacters__.register(name, value)
        # Merge inherited metacharacters
        merged_metacharacters = ProtodescriptorRegistry(cls.__merged_metadescriptors__)
        base_metacharacters = base.metacharacters("merged")
        merged_metacharacters.update(base_metacharacters)
        for trait in cls.__traits__:
            trait_metacharacters = trait.metacharacters("merged")
            merged_metacharacters.update(trait_metacharacters)
        merged_metacharacters.update(cls.__metacharacters__)
        cls.__merged_metacharacters__ = merged_metacharacters
        # TODO: validate metacharacters against archetype definition
        # Compilations and ast start empty and are filled when the declaring module is evaluated by the AML compiler  # noqa
        cls.__compilations__: dict[str, dict] = {}
        cls.__ast__ = None
        # Provisionally, new types are assigned the prototype role, but this may be overridden in decorators # noqa
        cls.__role__ = TypeRole.PROTOTYPE

    def __call__(cls, *args, **kwargs):
        raise TypeError("Archetypes, traits and prototypes cannot be instantiated directly.")

    def _normalize_traits(cls, *traits: Trait) -> tuple[Trait, ...]:
        """Establishes a normalized and ordered list of traits assignable to __merged_traits__.

        This method merges the supplied traits with those of the base type.
        It ensures that all traits conform to the archetype, and then creates a normalized list.
        The normalization establishes a consistent ordering of traits based on specialization and
        implementation hierarchy, and removes duplicates as well as unnecessary parents that are
        already included via inherited traits.
        """
        archetype = cls.__archetype__
        # Containers for the ordered traits and visited set
        order = []
        visited = set()

        def visit(T: Trait):
            """Recursively visit traits to build ordered list."""
            if T in visited:
                return
            # First, verifiy that the trait conforms to the archetype
            if not conforms(T, archetype):
                msg = f"Trait {T.__name__} does not conform to archetype {archetype.__name__}."
                raise TypeError(msg)
            # Ensure specialization parent comes first
            if T.supertrait is not None:
                visit(T.supertrait)
            # Then, ensure implemented traits come first, in declared order
            for U in T.__traits__:
                visit(U)
            visited.add(T)
            order.append(T)

        # Visit traits from base type first, then from supplied traits
        base: prototype = cls.__bases__[0]
        for T in base.traits("merged"):
            visit(T)
        for T in traits:
            visit(T)

        return tuple(order)

    @property
    def archetype(cls) -> Archetype:
        """The archetype implemented by this type."""
        return cls.__archetype__

    @property
    def basetype(cls) -> "prototype":
        """The base class of this type."""
        return cls.__bases__[0]

    @property
    def supertrait(cls) -> Trait | None:
        """The supertrait of this trait, if any."""
        if cls.__role__ != TypeRole.TRAIT:
            raise TypeError("Only trait types have a supertrait.")
        base: prototype = cls.__bases__[0]
        if base.__role__ == TypeRole.TRAIT:
            return cast(Trait, base)
        return None

    def implements(cls, metatype: Archetype | Trait) -> bool:
        """Checks if this type implements the specified archetype or trait."""
        if is_archetype(metatype):
            return metatype in cls.__archetype__.mro()

        if not is_trait(metatype):
            raise TypeError(f"Expected an Archetype or Trait type, got {metatype}.")

        for trait in cls.__merged_traits__:
            # direct match
            if trait is metatype:
                return True
            # otherwise walk specialization chain
            parent = trait.supertrait
            while parent is not None:
                if parent is metatype:
                    return True
                parent = parent.supertrait

        return False

    def traits(cls, view: Literal["local", "merged"]) -> tuple[Trait, ...]:
        """The traits implemented by this type.

        Args:
            view: Specifies which traits to return:
                - "local": resolved list of traits implemented on top of the archetype
                - "merged": resolved list of traits implemented by this type and its archetype
        """
        if view == "local":
            return cls.__traits__
        elif view == "merged":
            return cls.__merged_traits__
        else:
            msg = f"Invalid view '{view}'. Expected 'local' or 'merged'."
            raise ValueError(msg)

    def metadescriptors(cls, view: Literal["local", "merged", "resolved"]) -> MetadescriptorRegistry:  # noqa
        """The metadescriptors of this type.

        Args:
            view: Specifies which metadescriptors to return:
                - "local": metadescriptors declared directly on this type
                - "merged": metadescriptors declared on this type and inherited from archetype
                and traits
                - "resolved": metadescriptors after applying resolution rules, if any
        """
        if view == "local":
            return cls.__metadescriptors__.copy()
        elif view == "merged":
            return cls.__merged_metadescriptors__.copy()
        elif view == "resolved":
            return NotImplemented
        else:
            msg = f"Invalid view '{view}'. Expected 'local', 'merged', or 'resolved'."
            raise ValueError(msg)

    def metacharacters(cls, view: Literal["local", "merged", "resolved"]) -> ProtodescriptorRegistry:  # noqa
        """The metacharacters of this type.

        Args:
            view: Specifies which metacharacters to return:
                - "local": metacharacters declared directly on this type
                - "merged": metacharacters declared on this type and inherited from archetype and
                traits
                - "resolved": metacharacters after applying resolution rules, particularly
                default bindings, if any
        """
        if view == "local":
            return cls.__metacharacters__.copy()
        elif view == "merged":
            return cls.__merged_metacharacters__.copy()
        elif view == "resolved":
            return NotImplemented
        else:
            msg = f"Invalid view '{view}'. Expected 'local', 'merged', or 'resolved'."
            raise ValueError(msg)

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
