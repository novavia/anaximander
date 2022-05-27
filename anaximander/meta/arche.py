from abc import ABC
from contextlib import contextmanager
from typing import Any, ClassVar, Mapping, Optional

from pydantic import BaseModel
from sortedcontainers import SortedDict

from ..descriptors.base import DescriptorRegistry, SettingRegistry
from ..descriptors.methods import Metamethod, Metaproperty
from ..descriptors.typespec import TypeSpec, typespec
from ..descriptors.schema import MetadataSchema, OptionSchema


class Arche(ABC):
    """Base class for Anaximander archetypes."""

    # Archetype attributes
    # Note: basetype and metatype are respectively set to nxobject and nxtype
    basetype: ClassVar[type]
    metatype: ClassVar[type[type]]

    # Registries created from descriptor interpretation
    # traits and metacharacters are implemented as properties, because
    # they are populated after the Archetype has been initialized, and
    # they further must look up parent classes to collect inherited
    # registrations.
    descriptors: ClassVar[DescriptorRegistry]
    metadata: ClassVar[MetadataSchema] = MetadataSchema()
    options: ClassVar[OptionSchema] = OptionSchema()
    traits: ClassVar[dict[str, type]] = dict()
    metacharacters: ClassVar[tuple[str]] = tuple()
    _metacharacters: ClassVar[list[str]] = list()
    metaproperties: ClassVar[tuple[Metaproperty]] = tuple()
    metamethods: ClassVar[tuple[Metamethod]] = tuple()
    metamorphisms: ClassVar[dict[TypeSpec, type["Arche"]]] = SortedDict()

    # Type registries
    _types: ClassVar[dict[TypeSpec, type]] = dict()
    _traits: ClassVar[dict[str, type]] = dict()
    _writable: ClassVar[bool] = False

    # Base is a base type for subclassing archetypes declaratively rather
    # than programatically in a way that is more code-checker-friendly.
    # Note that this is *not* the basetype, but rather a class that is
    # directly derived from it, though run though the archetype's metatype.
    # In other words, Base could be written as:

    # class Base(cls.basetype, metaclass=cls.metatype):
    #     pass
    Base: ClassVar[type]

    def __new__(arc, *args, **kwargs):
        # If keyword arguments that match metdata are supplied,
        # these are interpreted to be turned into a type specification
        metadata_keys = set(arc.metadata.fields) & set(kwargs)
        metadata = {k: v for k, v in kwargs.items() if k in metadata_keys}
        typespec_ = arc.__typespec_interperter__(**metadata)
        archetype: type[Arche] = typespec_.archetype
        try:
            type_ = archetype._types[typespec_]
        except KeyError:
            type_ = archetype.__subtype__(typespec_, **kwargs)
        return type_(*args, **kwargs)

    @classmethod
    def __typespec_interperter__(arc, *metacharacters, **metadata) -> TypeSpec:
        """This method interprets resolves metacharacters and type parameters.

        This method basiscally allows a certain level of automation in equipping
        types and objects with features. Two use cases motivate its existence:
        * If a Model type declares a spatial field, then it can be made to
        inherit a spatial trait without requiring an explicit declaration
        * The other use case is the ability to add traits or delegate object
        creation to another archetype in response to object parameters -for
        instance, the scope of a data table influences its properties and methods.

        The method first runs metaproperties, which returns a tuple of
        metacharacters that are merged with the metacharacters already
        supplied in the method call.
        Based on this, we then evaluate whether there is a metamorphic
        archetype that match the specs, and if so delegate the remainder of
        the interpretation to that archetype.
        Otherwise, we resolve the metacharacters ordering and return a
        complete TypeSpec, including the archetype.
        """
        error_msg = (
            f"Improper type specifications {metacharacters} "
            + f"and {dict(metadata)} for {arc}, which accepts metacharacters "
            + f"{arc.metacharacters} and keyword arguments {list(arc.metadata.fields)}"
        )
        # First we validate the inputs
        try:
            assert set(metacharacters) <= set(arc.metacharacters)
            assert set(metadata) <= set(arc.metadata.fields)
        except AssertionError:
            raise TypeError(error_msg)
        metacharacters = set(metacharacters) | arc.__run_metaproperties__(**metadata)
        # We then remove superfluous metacharacters implemented by the basetype
        basetype_metacharacters = getattr(arc.basetype, "metacharacters")
        metacharacters -= basetype_metacharacters
        # To normalize the spec, we also remove the metadata that is set by the basetype
        type_metadata = arc.basetype.__type_metadata__(**metadata).dict(
            by_alias=True, exclude_unset=True, exclude_defaults=True
        )
        spec_metadata = {}
        basetype_metadata = getattr(arc.basetype, "_metadata").dict(by_alias=True)
        for k, v in type_metadata.items():
            try:
                assert v == basetype_metadata[k]
            except (KeyError, AssertionError):
                spec_metadata[k] = v
            else:
                continue
        provisional_spec = typespec(arc, *metacharacters, **spec_metadata)
        for spec, archetype in reversed(arc.metamorphisms.items()):
            if provisional_spec >= spec:
                return archetype.__typespec_interperter__(*metacharacters, **metadata)
        return provisional_spec

    @classmethod
    def __run_metaproperties__(arc, **metadata) -> set[str]:
        """This is a customizable method that computes metacharacters.

        Keyword arguments supplied to this method must be part of the archetype's
        metadata. A tuple of strings is returned, which are interpreted as metacharacters.
        This is used to fully specify subtypes that may inherit from traits as a result.
        """
        basetype = arc.basetype
        basetype_metadata: BaseModel = getattr(basetype, "_metadata")
        metadata = basetype_metadata.dict(by_alias=True) | metadata
        metacharacters = set()
        for metaprop in arc.metaproperties:
            if metaprop(**metadata):
                metacharacters.update(metaprop.metacharacters)
        return metacharacters

    @classmethod
    def __subtype__(
        arc,
        spec: TypeSpec,
        basetype: Optional[type] = None,
        type_name: Optional[str] = None,
        register_type: bool = True,
        overtype: bool = False,
        **kwargs,
    ):
        """A programmatic call to the metatype to return a new subtype.

        This method expects a normalized TypeSpec object, meaning that it
        must be run through __typespec_interpreter__ or guarantee that doing
        so would return the same specification. The spec's archetype must
        match cls.
        The reason for this constraint is to avoid duplication of efforts,
        since feeder methods already run __typespec_interpreter__.
        """
        try:
            assert spec.archetype is arc
        except AssertionError:
            msg = f"Incorrect specification {spec} supplied to {arc}"
            raise TypeError(msg)
        type_name = type_name or arc.__name__
        basetype = basetype or arc.basetype
        subtype: type[basetype] = arc.__subtype_new__(
            spec, type_name, basetype, **kwargs
        )
        arc.metatype.__init__(
            subtype,
            type_name,
            (basetype,),
            {},
            register_type=register_type,
            overtype=overtype,
            metacharacters=spec.metacharacters,
            **kwargs,
        )
        return subtype

    @classmethod
    def __subtype_new__(
        arc,
        spec: TypeSpec,
        type_name: Optional[str] = None,
        basetype: Optional[type] = None,
        namespace: Optional[Mapping] = None,
        **kwargs: Any,
    ):
        """Generates a new subtype, pre-initialization.

        This method is called by __new__ methods of both the Archetype and
        nxtype metaclasses to generate a new subtype.
        This method expects a normalized TypeSpec object, meaning that it
        must be run through __typespec_interpreter__ or guarantee that doing
        so would return the same specification. The spec's archetype must
        match cls.
        The reason for this constraint is to avoid duplication of efforts,
        since feeder methods already run __typespec_interpreter__.
        This method does not extract metadescriptors, and hence should not
        be called with a non-empty namespace -with the exception of the call
        made by nxtype.__new__.
        """
        type_name = type_name or arc.__name__
        if basetype is None:
            basetype = arc.basetype
        elif not issubclass(basetype, arc):
            msg = f"Cannot subtype {arc} with unrelated base {basetype}"
            raise TypeError(msg)
        try:
            assert spec.archetype is arc
        except AssertionError:
            msg = f"Incorrect specification {spec} supplied to {arc}"
            raise TypeError(msg)
        namespace = namespace or {}

        # We validate the keyword arguments
        unknown_kwargs = set(kwargs) - getattr(basetype, "__kwargs__", set())
        if unknown_kwargs:
            msg = f"Unknown keyword arguments {unknown_kwargs} in new {arc.__name__} creation"
            raise TypeError(msg)

        # Then we resolve the metaclass and parent classes
        # Traits are extracted in order to respect method resolution order
        # First, traits declared from derived archetypes take precedence over
        # those declared from parent archetypes. Second, within the traits
        # associated with a given archetype, the more recently declared and
        # registered traits take precedence in bases ordering.
        # This relies on the traits property defined by Archetype
        admissible_traits = getattr(arc, "traits", {})
        metacharacters = spec.metacharacters
        traits = [
            t
            for m, t in admissible_traits.items()
            if m in metacharacters and t not in basetype.mro()
        ]
        bases = tuple(reversed(traits)) + (basetype,)
        metaclass = arc.metatype

        # New descriptor declarations are collected from the namespace
        basetype_descriptors = getattr(basetype, "__descriptors__")
        descriptor_registry = DescriptorRegistry(parent=basetype_descriptors)
        namespace["__descriptors__"] = descriptor_registry
        descriptor_registry.collect(namespace, strip=True)
        # And from traits as well
        for trait in traits:
            trait_descriptors = getattr(trait, "__descriptors__", {})
            descriptor_registry.collect(trait_descriptors)

        # Archetypical field settings are collected from the namespace and kwargs
        basetype_settings = getattr(basetype, "__settings__")
        settings_registry = SettingRegistry(
            descriptor_registry, parent=basetype_settings
        )
        namespace["__settings__"] = settings_registry
        for trait in traits:
            trait_settings = getattr(trait, "__settings__", {})
            settings_registry.collect(trait_settings)
        settings_registry.collect(namespace, strip=True)
        settings_registry.collect(kwargs)

        new_subtype: type[basetype] = type.__new__(
            metaclass, type_name, bases, namespace
        )
        return new_subtype

    @contextmanager
    @classmethod
    def writable(arc):
        """A context manager to make the archetype temporarily modifiable.

        This is used for registering resources, and intended for metaprogramming
        purposes only.
        """
        type.__setattr__(arc, "_writable", True)
        try:
            yield arc
        finally:
            arc._writable = False


class ArcheBase(Arche):
    pass


Arche.Base = ArcheBase
