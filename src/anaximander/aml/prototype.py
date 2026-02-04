# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Define the prototype metaclass and prototype-level registries for AML.

Prototypes are the concrete AML classes created by the ``prototype`` metaclass.
They aggregate declarators from archetypes and traits, normalize inheritance,
and expose merged registries for declarations and bindings.

This module implements the registries used by prototypes and the metaclass
behavior that merges traits, validates declarators, and establishes runtime
metadata used by compilers and the digital twin interface.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports


from contextlib import contextmanager
from enum import Enum
from functools import update_wrapper
from typing import Any, Callable, ClassVar, Literal, TypeGuard, cast

from annotationlib import Format, get_annotations

from ..utils.funcs import (
    type_name_to_collection_name,
    unwrap_classvar_type,
    unwrap_optional_type,
)
from ..utils.yaml import nx_register_representer, nx_set_ignore_aliases
from .declarative import DeclarativeNamespace, declarative
from .declarators import (
    AnnotatableDeclarator,
    ConstructorDeclarator,
    DeclarativeProtocol,
    Declarator,
    FieldProtodescriptor,
    MetadataDeclarator,
    Metadescriptor,
    MetaValidator,
    NxFieldDeclarator,
    OptionDeclarator,
    PrototypeValidator,
    SchemaDeclarator,
)
from .diagnostics import (
    PROTOTYPE_DIAGNOSTICS,
    AnnotatableDeclaratorUnnamed,
    ArchetypeTraitExpected,
    DeclaratorAnnotationMissing,
    DeclaratorNotAllowedInArchetype,
    DiagnosticBag,
    InvalidPrototypeBase,
    InvalidViewSelection,
    MetadataDomainBindingForbidden,
    MetadataNotDeclared,
    MultipleInheritanceUnsupported,
    PrototypeInstantiationForbidden,
    TraitConformanceViolation,
    TraitSupertraitForbidden,
)
from .registries import (
    BindingRegistry,
    DeclaratorRegistry,
    MultiBindingRegistry,
    MultiDeclaratorRegistry,
)

# endregion

# =============================================================================
# Registries
# =============================================================================
# region Registries


class PrototypeDeclaratorRegistry(MultiDeclaratorRegistry):
    """Registry for prototype declarators."""
    __handles__ = {"metadata", "nxfield", "option", "metavalidator", "field", "schema", "constructor"}  # noqa
    __namespaces__ = {"domain", "nx"}

    def __init__(self):
        super().__init__()
        # Initialize registries for each namespace
        metadata = DeclaratorRegistry[MetadataDeclarator]()
        nxfield = DeclaratorRegistry[NxFieldDeclarator]()
        option = DeclaratorRegistry[OptionDeclarator]()
        metavalidator = DeclaratorRegistry[MetaValidator]()
        field = DeclaratorRegistry[FieldProtodescriptor]()
        schema = DeclaratorRegistry[SchemaDeclarator]()
        constructor = DeclaratorRegistry[ConstructorDeclarator]()
        self._registries = {
            "metadata": metadata,
            "nxfield": nxfield,
            "option": option,
            "metavalidator": metavalidator,
            "field": field,
            "schema": schema,
            "constructor": constructor,
        }
        self._namespaces = {
            "domain": dict(),
            "nx": dict(),
        }

    @property
    def metadata(self) -> DeclaratorRegistry[MetadataDeclarator]:
        """Returns the metadata metadescriptor registry."""
        return cast(DeclaratorRegistry[MetadataDeclarator], self._registries["metadata"])

    @property
    def nxfield(self) -> DeclaratorRegistry[NxFieldDeclarator]:
        """Returns the nxfield metadescriptor registry."""
        return cast(DeclaratorRegistry[NxFieldDeclarator], self._registries["nxfield"])

    @property
    def option(self) -> DeclaratorRegistry[OptionDeclarator]:
        """Returns the option metadescriptor registry."""
        return cast(DeclaratorRegistry[OptionDeclarator], self._registries["option"])

    @property
    def metavalidator(self) -> DeclaratorRegistry[MetaValidator]:
        """Returns the metavalidator metadescriptor registry."""
        return cast(DeclaratorRegistry[MetaValidator], self._registries["metavalidator"])

    @property
    def field(self) -> DeclaratorRegistry[FieldProtodescriptor]:
        """Returns the field protodescriptor registry."""
        return cast(DeclaratorRegistry[FieldProtodescriptor], self._registries["field"])

    @property
    def schema(self) -> DeclaratorRegistry[SchemaDeclarator]:
        """Returns the schema declarator registry."""
        return cast(DeclaratorRegistry[SchemaDeclarator], self._registries["schema"])

    @property
    def constructor(self) -> DeclaratorRegistry[ConstructorDeclarator]:
        """Returns the constructor declarator registry."""
        return cast(DeclaratorRegistry[ConstructorDeclarator], self._registries["constructor"])

    @classmethod
    def handle(cls, declarator: Declarator) -> str | None:
        """Returns the registry handle for the given declarator, or None if not found."""
        match declarator:
            case MetadataDeclarator():
                return "metadata"
            case NxFieldDeclarator():
                return "nxfield"
            case OptionDeclarator():
                return "option"
            case MetaValidator():
                return "metavalidator"
            case FieldProtodescriptor():
                return "field"
            case SchemaDeclarator():
                return "schema"
            case ConstructorDeclarator():
                return "constructor"
            case _:
                return None

    @classmethod
    def namespace(cls, declarator: Declarator) -> str | None:
        """Returns the registry namespace for the given declarator."""
        match declarator:
            case MetadataDeclarator():
                if declarator.domain is True:
                    return "domain"
                else:
                    return "nx"
            case Metadescriptor():
                return "nx"
            case FieldProtodescriptor() | ConstructorDeclarator():
                return "domain"
            case _:
                return None


class PrototypeBindingRegistry(MultiBindingRegistry):
    """Registry for prototype bindings."""
    __handles__ = {"metadata", "nxfield", "option", "data"}
    __auto_handles__ = {"metadata", "data"}

    def __init__(self, declarators: PrototypeDeclaratorRegistry):
        super().__init__(declarators)
        metadata = BindingRegistry(_declarators=declarators.metadata)
        nxfield = BindingRegistry(_declarators=declarators.nxfield)
        option = BindingRegistry(_declarators=declarators.option)
        data = BindingRegistry(_declarators=declarators.field)
        self._registries = {
            "metadata": metadata,
            "nxfield": nxfield,
            "option": option,
            "data": data,
        }

    @property
    def metadata(self) -> BindingRegistry[MetadataDeclarator]:
        """Returns the metadata binding registry."""
        return cast(BindingRegistry[MetadataDeclarator], self._registries["metadata"])

    @property
    def nxfield(self) -> BindingRegistry[NxFieldDeclarator]:
        """Returns the nxfield binding registry."""
        return cast(BindingRegistry[NxFieldDeclarator], self._registries["nxfield"])

    @property
    def option(self) -> BindingRegistry[OptionDeclarator]:
        """Returns the option binding registry."""
        return cast(BindingRegistry[OptionDeclarator], self._registries["option"])

    @property
    def data(self) -> BindingRegistry[FieldProtodescriptor]:
        """Returns the data binding registry."""
        return cast(BindingRegistry[FieldProtodescriptor], self._registries["data"])

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


class prototypeProtocol(DeclarativeProtocol):
    """Protocol for prototype classes."""
    __archetype__: ClassVar["Archetype"]
    __traits__: ClassVar[tuple["Trait", ...]]
    __merged_traits__: ClassVar[tuple["Trait", ...]]
    __declarators__: ClassVar[PrototypeDeclaratorRegistry]
    __bindings__: ClassVar[PrototypeBindingRegistry]
    __merged_declarators__: ClassVar[PrototypeDeclaratorRegistry]
    __merged_bindings__: ClassVar[PrototypeBindingRegistry]
    __compilations__: ClassVar[dict[str, dict]]
    __diagnostics__: ClassVar["DiagnosticBag"]

    @classmethod
    def implements(cls, metatype: "Archetype | Trait") -> bool:
        """Checks if this type implements the specified archetype or trait."""
        ...

    @classmethod
    def declarators(cls, view: Literal["local", "merged", "resolved"]) -> PrototypeDeclaratorRegistry:  # noqa
        """The declarators of this type."""
        ...

    @classmethod
    def bindings(cls, view: Literal["local", "merged", "resolved"]) -> PrototypeBindingRegistry:  # noqa
        """The bindings of this type."""
        ...


class ArchetypeProtocol(prototypeProtocol):
    """Protocol for archetype classes."""
    __role__: ClassVar[Literal[TypeRole.ARCHETYPE]]
    __declarator_types__: ClassVar[set[type[Declarator]]]

    @classmethod
    def __validate_declarators__(cls, owner: "prototype") -> None:
        """Validate declarators in the context of an owner."""
        # Default implementation is a no-op; archetypes/traits can override.
        ...

Archetype = type[ArchetypeProtocol]

def is_archetype(cls: type[Any]) -> TypeGuard[Archetype]:
    """Type guard to check if a class is an Archetype."""
    if cls is Arche:
        return True
    return (isinstance(cls, prototype) and getattr(cls, "__role__", None) == TypeRole.ARCHETYPE)


class TraitProtocol(prototypeProtocol):
    """Protocol for trait classes."""
    __role__: ClassVar[Literal[TypeRole.TRAIT]]
    supertrait: ClassVar["Trait | None"]

    @classmethod
    def __validate_declarators__(cls, owner: "prototype") -> None:
        """Validate declarators in the context of an owner."""
        ...

Trait = type[TraitProtocol]

def is_trait(cls: type[Any]) -> TypeGuard[Trait]:
    """Type guard to check if a class is a Trait."""
    return (isinstance(cls, prototype) and getattr(cls, "__role__", None) == TypeRole.TRAIT)

def conforms(trait: Trait, archetype: Archetype) -> bool:
    """Checks if a trait conforms to the specified archetype."""
    if not is_trait(trait):
        ArchetypeTraitExpected.report(f"Expected a Trait type, got {trait!r}.")
    if not is_archetype(archetype):
        ArchetypeTraitExpected.report(f"Expected an Archetype type, got {archetype!r}.")
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
    _declarators: ClassVar[PrototypeDeclaratorRegistry] = PrototypeDeclaratorRegistry()
    _bindings: ClassVar[PrototypeBindingRegistry] = PrototypeBindingRegistry(_declarators)
    __declarator_types__: ClassVar[set[type[Declarator]]] = {Metadescriptor}
    __diagnostics__: ClassVar["DiagnosticBag"]

    def __new__(cls, *args, **kwargs):
        PrototypeInstantiationForbidden.report(
            "Archetypes, traits and prototypes cannot be instantiated directly."
        )
        return super().__new__(cls)

    @classmethod
    def traits(cls, view: Literal["local", "merged", "total"]) -> tuple[Trait, ...]:
        """The traits of this archetype."""
        if isinstance(cls, prototype):
            if view == "total":
                view = "merged"
            return prototype.traits(cls, view)
        return ()

    @classmethod
    def declarators(cls, view: Literal["local", "merged", "resolved"]) -> PrototypeDeclaratorRegistry:  # noqa
        """The declarators of this archetype."""
        if isinstance(cls, prototype):
            return prototype.declarators(cls, view)
        return cls._declarators.copy()
    @classmethod
    def bindings(cls, view: Literal["local", "merged", "resolved"]) -> PrototypeBindingRegistry:  # noqa
        """The bindings of this archetype."""
        if isinstance(cls, prototype):
            return prototype.bindings(cls, view)
        return cls._bindings.copy()

    @classmethod
    def __validate_declarators__(cls, owner: "prototype") -> None:
        """Validate declarators in the context of an owner."""

Arche.__archetype__ = cast(Archetype, Arche)


class prototype(declarative):
    """Metaclass for declarative types -archetypes, traits and prototypes."""
    __role__: TypeRole
    __archetype__: Archetype  # The prototype's archetype
    __traits__: tuple[Trait, ...]  # The prototype's traits implemented on top of its archetype
    __merged_traits__: tuple[Trait, ...]  # The prototype's merged traits, including inherited ones
    __declarators__: PrototypeDeclaratorRegistry  # The prototype's locally declared declarators
    __bindings__: PrototypeBindingRegistry  # The prototype's locally declared bindings
    __merged_declarators__: PrototypeDeclaratorRegistry  # The prototype's merged declarators
    __merged_bindings__: PrototypeBindingRegistry  # The prototype's merged bindings
    __compilations__: dict[str, dict]  # Holds compilation target handles and parameters
    __diagnostics__: DiagnosticBag  # Holds prototype-scoped diagnostics

    def __new__(mcls, name, bases, namespace: DeclarativeNamespace, traits=(), **metadata):
        base = bases[0]
        if not (isinstance(base, prototype) or base is Arche):
            InvalidPrototypeBase.report(f"Base class {base} is not a valid AML prototype base.")
        if any(isinstance(base, prototype) for base in bases[1:]):
            MultipleInheritanceUnsupported.report(
                "Prototypes do not support multiple inheritance."
            )
        cls = super().__new__(mcls, name, bases, namespace)
        # Phase: set type annotations on annotatable declarators.
        with _prototype_diagnostics(cls):
            cls.__set_type_annotations__()

        return cls

    def __init__(cls, name, bases, namespace: DeclarativeNamespace, traits=(), **metadata):
        super().__init__(name, bases, namespace, **metadata)
        # Phase: resolve archetype and trait relationships.
        with _prototype_diagnostics(cls):
            # Set the base archetype.
            base: prototype | type[Arche]= bases[0]
            base_archetype: Archetype = getattr(base, "__archetype__")
            archetype = cls.__archetype__ = base_archetype
            # Next we normalize and set traits.
            # Merged traits include those of the base type.
            cls.__merged_traits__ = prototype._normalize_traits(cls, *traits)
            # Local traits are those that concretely specialize the base.
            base_traits = base.traits("merged")
            cls.__traits__ = tuple(T for T in cls.__merged_traits__ if T not in base_traits)
            # Validate that all declarations conform to the archetype.
            allowed_declarator_types = tuple(getattr(archetype, "__declarator_types__", set()))
            for declarator in cls.__raw_declarators__.values():
                if not isinstance(declarator, allowed_declarator_types):
                    DeclaratorNotAllowedInArchetype.report(
                        f"{declarator!r} is not allowed in archetype {archetype!r}."
                    )
        # Phase: register declarators and bindings.
        with _prototype_diagnostics(cls):
            # Set local declarators.
            cls.__declarators__ = PrototypeDeclaratorRegistry()
            for declarator in cls.__raw_declarators__.values():
                if isinstance(declarator, Declarator):
                    cls.__declarators__.register(declarator.name, declarator)
            # Merge inherited declarators.
            merged_declarators = base.declarators("merged")
            for trait in cls.__traits__:
                trait_declarators = trait.declarators("merged")
                merged_declarators.update(trait_declarators)
            merged_declarators.update(cls.__declarators__)
            cls.__merged_declarators__ = merged_declarators
            # Set local bindings.
            cls.__bindings__ = PrototypeBindingRegistry(declarators=cls.__merged_declarators__)
            for name, value in cls.__raw_bindings__.items():
                if "." in name:
                    handle, binding_name = name.split(".", 1)
                    cls.__bindings__.register(binding_name, value, handle=handle)
                else:
                    cls.__bindings__.register(name, value)
            # And additionally set metadata passed in the class header.
            # Only non-domain metadata can be set this way.
            for name, value in metadata.items():
                declarator = merged_declarators.metadata.get(name)
                if declarator is None or not isinstance(declarator, MetadataDeclarator):
                    MetadataNotDeclared.report(
                        f"Metadata '{name}' is not declared for prototype {cls!r}."
                    )
                elif declarator.domain is True:
                    MetadataDomainBindingForbidden.report(
                        (f"Cannot set domain metadata '{name}' in class header of "
                         f"{cls!r}.")
                    )
                cls.__bindings__.register(name, value, handle="metadata")
            # Merge local and inherited bindings.
            merged_bindings = PrototypeBindingRegistry(cls.__merged_declarators__)
            base_bindings = base.bindings("merged")
            merged_bindings.update(base_bindings)
            for trait in cls.__traits__:
                trait_bindings = trait.bindings("merged")
                merged_bindings.update(trait_bindings)
            merged_bindings.update(cls.__bindings__)
            cls.__merged_bindings__ = merged_bindings
        # Binding validation and registered validators run at module finalization.
        # Compilations and ast start empty and are filled when the declaring module is evaluated by the AML compiler  # noqa
        cls.__compilations__: dict[str, dict] = {}
        cls.__ast__ = None
        # Provisionally, new types are assigned the prototype role, but this may be overridden in decorators # noqa
        cls.__role__ = TypeRole.PROTOTYPE

    @staticmethod
    def validator():
        """Create a prototype validator as a decorator."""
        def decorator(fn: Callable[["prototype"], bool]) -> PrototypeValidator:
            declarator = PrototypeValidator(callable=fn, doc=fn.__doc__)  # type: ignore[abstract]
            update_wrapper(declarator, fn, updated=())  # type: ignore[arg-type]
            return declarator

        return decorator

    @property
    def _bindable_domain_names(cls) -> set[str]:
        """Returns the set of names that subclasses can bind within the domain namespace.

        These include domain metadata and fields that are class variables.
        """
        bindable_domain_names = set()
        # Domain metadata
        for md in cls.__merged_declarators__.metadata.values():
            if md.domain_bindable:
                bindable_domain_names.add(md.name)
        # Class variable protodescriptors
        for fd in cls.__merged_declarators__.field.values():
            if fd.domain_bindable:
                bindable_domain_names.add(fd.name)
        return bindable_domain_names

    def __call__(cls, *args, **kwargs):
        PrototypeInstantiationForbidden.report(
            "Archetypes, traits and prototypes cannot be instantiated directly."
        )
        return None

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
                TraitConformanceViolation.report(trait=T, archetype=archetype)
                return
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
            TraitSupertraitForbidden.report("Only trait types have a supertrait.")
        base: prototype = cls.__bases__[0]
        if base.__role__ == TypeRole.TRAIT:
            return cast(Trait, base)
        return None

    def implements(cls, metatype: Archetype | Trait) -> bool:
        """Checks if this type implements the specified archetype or trait."""
        if is_archetype(metatype):
            return metatype in cls.__archetype__.mro()

        if not is_trait(metatype):
            ArchetypeTraitExpected.report(
                f"Expected an Archetype or Trait type, got {metatype!r}."
            )

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
            InvalidViewSelection.report(
                f"Invalid view '{view}'. Expected 'local' or 'merged'."
            )

    def declarators(cls, view: Literal["local", "merged", "resolved"]) -> PrototypeDeclaratorRegistry:  # noqa
        """The declarators of this type.

        Args:
            view: Specifies which declarators to return:
                - "local": declarators declared directly on this type
                - "merged": declarators declared on this type and inherited from archetype
                and traits
                - "resolved": declarators after applying resolution rules, if any
        """
        if view == "local":
            return cls.__declarators__.copy()
        elif view == "merged":
            return cls.__merged_declarators__.copy()
        elif view == "resolved":
            return NotImplemented
        else:
            InvalidViewSelection.report(
                f"Invalid view '{view}'. Expected 'local', 'merged', or 'resolved'."
            )

    def bindings(cls, view: Literal["local", "merged", "resolved"]) -> PrototypeBindingRegistry:  # noqa
        """The bindings of this type.

        Args:
            view: Specifies which bindings to return:
                - "local": bindings declared directly on this type
                - "merged": bindings declared on this type and inherited from archetype and
                traits
                - "resolved": bindings after applying resolution rules, particularly
                default bindings, if any
        """
        if view == "local":
            return cls.__bindings__.copy()
        elif view == "merged":
            return cls.__merged_bindings__.copy()
        elif view == "resolved":
            return NotImplemented
        else:
            InvalidViewSelection.report(
                f"Invalid view '{view}'. Expected 'local', 'merged', or 'resolved'."
            )

    @staticmethod
    def __unwrap_optional_type__(type_hint: Any) -> tuple[Any, bool]:
        """Detect optional annotations and return base type with a nullable flag."""
        return unwrap_optional_type(type_hint)

    @staticmethod
    def __unwrap_classvar_type__(type_hint: Any) -> tuple[Any, bool]:
        """Detect ClassVar annotations and return base type with a classvar flag."""
        return unwrap_classvar_type(type_hint)

    def __set_type_annotations__(cls):
        """Sets type annotations on typed protodescriptors."""
        # Get annotations defined on this class
        annotation_values = get_annotations(cls, format=Format.VALUE)
        annotation_strings = get_annotations(cls, format=Format.STRING)
        # Fetch annotatable declarators declared on this class
        annotatable_declarators = [
            declarator for declarator in cls.__raw_declarators__.values()
            if isinstance(declarator, AnnotatableDeclarator)
        ]
        # Set annotations on each annotatable declarator
        for declarator in annotatable_declarators:
            # This is a sentinel, as we expect all annotatable declarators to be named at this stage # noqa
            if (name := declarator.name) is None:
                AnnotatableDeclaratorUnnamed.report(
                    "Annotatable declarators must be named before annotation binding."
                )
            # Then we distinguish between declarators that were declared through assignment vs
            # those that relied on the DeclaratorHandle.declare method
            is_assigned = getattr(cls, name, None) is declarator
            if not is_assigned:
                continue
            # For assigned declarators, we enforce that an annotation must be present
            if name not in annotation_values and name not in annotation_strings:
                DeclaratorAnnotationMissing.report(
                    f"Missing annotation for declarator {declarator!r}."
                )
            annotation_value = annotation_values[name]
            annotation_string = annotation_strings[name]
            if isinstance(annotation_value, str):
                annotation = f'"{annotation_value}"'
                hint = None
                classvar = None
                nullable = None
            else:
                annotation = annotation_string
                hint, classvar = cls.__unwrap_classvar_type__(annotation_value)
                hint, nullable = cls.__unwrap_optional_type__(hint)
            # TODO: normalize to a prototype in case of model attributes
            # TODO: normalize to a metadata type in case of option / metacharacter
            declarator.__set_type__(annotation, hint, nullable, classvar, hint=hint)

    def __repr__(cls) -> str:
        return f"<{cls.__name__} {cls.__role__.value}>"

    @property
    def collection_name(cls):
        """A collection name using camel case and pluralization.

        This can be customized by passing metadata (#TODO).
        """
        return type_name_to_collection_name(cls.__name__)


@contextmanager
def _prototype_diagnostics(cls: prototype):
    """Set prototype diagnostics context for a single lifecycle phase."""
    # if not hasattr(cls, "__diagnostics__"):
    #     cls.__diagnostics__ = DiagnosticBag(cls)
    token = PROTOTYPE_DIAGNOSTICS.set(cls.__diagnostics__)
    try:
        yield
    finally:
        PROTOTYPE_DIAGNOSTICS.reset(token)


def _register_aml_yaml_prototypes() -> None:
    """Register YAML handlers for prototype references."""
    def _repr_type(dumper, obj: type):
        return dumper.represent_scalar("tag:yaml.org,2002:str", f"<{obj.__name__}>")

    def _repr_function(dumper, obj: object):
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", obj.__class__.__name__))
        return dumper.represent_scalar("tag:yaml.org,2002:str", f"<{name}>")

    def _repr_prototype(dumper, obj: prototype):
        return dumper.represent_scalar("tag:yaml.org,2002:str", repr(obj))

    nx_set_ignore_aliases(lambda value: isinstance(value, type))
    nx_register_representer(type, _repr_type)
    nx_register_representer(prototype, _repr_prototype)
    from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
    nx_register_representer(FunctionType, _repr_function)
    nx_register_representer(BuiltinFunctionType, _repr_function)
    nx_register_representer(MethodType, _repr_function)
    nx_register_representer(BuiltinMethodType, _repr_function)


_register_aml_yaml_prototypes()
