from abc import ABCMeta
from typing import Any, Generic, Optional, TypeVar

from sortedcontainers.sorteddict import SortedDict


from ..utilities.functions import inherited_property, typeproperty


from ..descriptors.typespec import TypeSpec
from ..descriptors.base import SettingRegistry
from ..descriptors.methods import Metamethod, Metaproperty
from ..descriptors.schema import MetadataSchema, OptionSchema

from .arche import Arche
from .nxmeta import nxmeta
from .nxobject import nxobject, nxtrait


T = TypeVar("T", bound=nxobject, covariant=True)


class Archetype(ABCMeta, Generic[T]):
    """The metaclass for archetypes."""

    # A key feature of this metaclass is that it ironically diverts subclasses
    # away from itself by default. When a type subtypes an archetype, the
    # archetype effectively calls its so-called metatype, which creates a
    # subclass of the base type. Note that for this reason the interface
    # must allow arbitrary arguments -they are unnecesary for generating
    # new archetypes (though that may change to meet a future functional need),
    # but they must be accepted to be passed through to the metatype.
    # The keyword __basetype__ in new class declarations acts as a sentinel
    # to distinguish between a situation in which the archetype is summoned
    # to generate a new subtype within its clade, and a situation in which a new
    # Archetype must be created, which is normally called programatically
    # with the @archetype decorator: in that case, the decorator supplies the
    # __basetype__ keyword argument in its call to Archetype.

    def __new__(
        mcl,
        name,
        bases,
        namespace,
        register_type: bool = True,
        overtype: bool = False,
        metacharacters: tuple[str] = (),
        **kwargs,
    ) -> type[T]:
        """This interface mimics nxtype's to produce concrete types.

        Actual archetypes are created with the __new_archetype__ method.
        """
        # Enforce that the basetype subclasses the metaclass' archetype
        archetype: type["Arche"] = bases[0]
        bases = (archetype.basetype,) + bases[1:]
        return archetype.metatype(
            name,
            bases,
            namespace,
            register_type=register_type,
            overtype=overtype,
            metacharacters=metacharacters,
            **kwargs,
        )

    @classmethod
    def __new_archetype__(
        mcl,
        name,
        bases,
        namespace,
        *,
        basetype: type[T],
        metatype: Optional[type[type]] = None,
    ) -> "Archetype[T]":
        return type.__new__(mcl, name, bases, namespace)

    @classmethod
    def _make_archetype(
        mcl, basetype: type[T], metatype: Optional[type[type]] = None
    ) -> "Archetype[T]":
        try:
            assert issubclass(basetype, nxobject)
            # basetype is not a Trait
            assert not isinstance(basetype, nxtrait)
        except AssertionError:
            msg = f"Cannot make new archetype from basetype {basetype}."
            raise TypeError(msg)
        name = basetype.__name__
        # The base archetype for the new archetype
        base = getattr(basetype, "archetype", Arche)
        if metatype is not None:
            try:
                assert issubclass(metatype, base.metatype)
            except AssertionError:
                msg = f"Invalid metaclass {metatype} supplied to create archetype from {basetype}"
                raise TypeError(msg)
        new_archetype = mcl.__new_archetype__(
            name, (base,), {}, basetype=basetype, metatype=metatype
        )
        Archetype.__init__(
            new_archetype, name, (base,), {}, basetype=basetype, metatype=metatype
        )
        return new_archetype

    def __init__(
        arc,
        name,
        bases,
        namespace,
        *,
        basetype: type[T],
        metatype: Optional[type[type]] = None,
    ):
        type.__setattr__(arc, "_writable", True)
        ABCMeta.__init__(arc, name, bases, namespace)
        # Sets basetype and a blank type registry
        arc.basetype = basetype
        arc._types = dict()
        # Extract the base archetype from the basetype
        base_archetype: type[Arche] = basetype.archetype
        base_metatype = metatype or base_archetype.metatype

        # Descriptors are extracted from the basetype
        arc.descriptors = descriptors = basetype.__descriptors__

        # Registries are built from descriptors

        # metadata
        metadata_fields = descriptors.fetch("metadata", recursive=True).values()
        metadata_parsers = list(
            descriptors.fetch("metadataparser", recursive=True).values()
        )
        metadata_validators = list(
            descriptors.fetch("metadatavalidator", recursive=True).values()
        )
        if len(metadata_parsers) > 1:
            msg = "Only one metadata parser can be specified"
            raise TypeError(msg)
        elif len(metadata_parsers) == 1:
            metadata_parser = metadata_parsers.pop().method
        else:
            metadata_parser = None
        metadata_validators = [v.method for v in metadata_validators]
        arc.metadata = MetadataSchema(
            fields=metadata_fields,
            parser=metadata_parser,
            validators=metadata_validators,
        )

        # options
        option_fields = descriptors.fetch("options", recursive=True).values()
        option_parser = list(descriptors.fetch("optionparser", recursive=True).values())
        option_validators = list(
            descriptors.fetch("optionvalidator", recursive=True).values()
        )
        if len(option_parser) > 1:
            msg = "Only one metadata parser can be specified"
            raise TypeError(msg)
        elif len(option_parser) == 1:
            option_parser = option_parser.pop().method
        else:
            option_parser = None
        option_validators = [v.method for v in option_validators]
        arc.options = OptionSchema(
            fields=option_fields, parser=option_parser, validators=option_validators
        )
        # traits
        arc._traits = base_archetype.traits.copy()
        # metaproperties
        metaproperties: tuple[Metaproperty] = tuple(
            descriptors.fetch("metaproperties", recursive=True).values()
        )
        arc.metaproperties = metaproperties
        # metamethods
        metamethods: tuple[Metamethod] = tuple(
            descriptors.fetch("metamethods", recursive=True).values()
        )
        arc.metamethods = metamethods
        # like traits, new metamorphisms are registered post-init
        arc.metamorphisms = SortedDict()
        # metatcharacters
        metacharacters = list(base_archetype.metacharacters)
        for metaproperty in arc.metaproperties:
            for mch in metaproperty.metacharacters:
                if not mch in metacharacters:
                    metacharacters.append(mch)
        arc._metacharacters = metacharacters

        # We set attributes of the archetype from the descriptors
        attributes = descriptors.fetch()
        for name_, descriptor in attributes.items():
            setattr(arc, name_, descriptor)
        settings: SettingRegistry = getattr(basetype, "__settings__")
        metadata = settings.fetch("metadata", recursive=True)
        options = settings.fetch("options", recursive=True)
        for name_ in metadata | options:
            setattr(arc, name_, getattr(basetype, name_))

        # And those of the basetype
        new_fields = descriptors.fetch("metadata", "options")
        for field in new_fields.values():
            field.set_attribute(basetype)

        # Now we create the metatype, responsible for creating subtypes
        metatype = arc.metatype = nxmeta(
            name + "Type", (base_metatype,), {}, archetype=arc
        )

        # The metadata and option models computed by the metaclass are passed
        # to the basetype
        basetype.__type_metadata__ = metatype.__type_metadata__
        basetype.__metadata__ = metatype.__metadata__
        basetype.__options__ = metatype.__options__

        # And we further add inherited properties for any typeproperty
        # declared in the metatype
        for name, attr in vars(arc.metatype).items():
            if isinstance(attr, typeproperty):
                setattr(basetype, name, inherited_property(attr))

        # And we set the admissible keyword arguments
        basetype.__type_kwargs__ = set(arc.metadata.fields | arc.options.fields)
        basetype.__kwargs__ = (
            getattr(basetype, "__kwargs__", set()) | basetype.__type_kwargs__
        )

        # We set the Base attribute, which is a direct derivative of the
        # basetype, but run through the metatype. This sets the .Base
        # property, which is currently the recommended way to subclass
        # archetypes declaratively, e.g.
        #
        # class MyObject(Object.Base): ...
        #
        # The MyObject class can be a concrete type, or it could be
        # decorated with @archetype decorator.
        # Note that this equivalent to:
        #
        # class MyObject(Object): ...
        #
        # which would be the preferred syntax, but the former enables static
        # checkers to correctly infer the type of MyObject, whereas the latter
        # does not.
        arc.Base: type[T] = arc.subtype()
        arc._writable = False

    def __subclasscheck__(arc: "Archetype", cls: type) -> bool:
        if not isinstance(cls, type):
            raise TypeError("issubclass() arg 1 must be a class")
        return {arc, arc.basetype} & set(cls.__mro__)

    @property
    def archetype(arc) -> "Archetype[T]":
        """The archetype itself, provided for compatibility with regular types."""
        return arc

    @property
    def traits(arc) -> dict[str, nxtrait]:
        """A mapping of registered metacharacters to mix-in traits, in method resolution order."""
        return arc._traits.copy()

    @property
    def metacharacters(arc) -> tuple[str]:
        """A tuple of admissible metacharacters, in declaration order."""
        return tuple(arc._metacharacters)

    def retrieve_type(arc: type[Arche], *metacharacters: str, **metadata) -> type[T]:
        """Retrieves a type from an archetype and subsequent specifications."""
        spec = arc.__typespec_interperter__(*metacharacters, **metadata)
        return arc._types[spec]

    def retrieve_type_from_key(arc, key, *, subtype=False) -> type[T]:
        """Retrieves a type from an archetype and a registration key.

        This method implements the __getitem__ functionality of archetypes.
        Key can be any object that can be converted to a set of metacharacters
        and a mapping of type parameters by implementing the __typespec__
        special method.
        Another possibility is if the archetype implements a single type
        parameter, call it 'param' for example sake, in which case the method
        is a shortcut, such that archetype.retrieve_type_from_key(x) is
        equivalent to archetype_retrieve_type(param=x).
        If the key cannot be interpreted, a ValueError is raised. If the key
        can be interpreted but no type can be retrieved, a KeyError is raised
        as one might expect. However if subtype is True, the method will
        instead attempt to create and return a type corresponding to key
        (equivalent to subtype).
        """
        method = arc.subtype if subtype else arc.retrieve_type
        try:
            metacharacters, type_parameters = getattr(key, "__typespec__")(arc)
        except AttributeError:
            pass
        else:
            return method(*metacharacters, **type_parameters)
        if len(params := arc.metatype.__unset_typespec_metadata__) == 1:
            spec = {params[0]: key}
            return method(**spec)
        else:
            msg = f"Could not interpret key {key} for type retrieval on {arc}"
            raise ValueError(msg)

    def subtype(
        arc: type[Arche],
        *metacharacters: str,
        basetype: Optional[type[nxobject]] = None,
        type_name: Optional[str] = None,
        register_type: bool = True,
        overtype: bool = False,
        **kwargs,
    ) -> type[T]:
        """Returns the specified subtype, either from cache or after creating it.

        There is an optional type_name parameter. If supplied, it is assumed that
        the programmer seeks to create a new type. The cache will still be
        looked up from the specifications, and if a type is retrieved whose name
        matches type_name, then the cached type is returned. If the names do not
        match, a new type with the desired name is returned per specifications.
        By default, newly created types are regularly registered with the archetype.
        This behavior can be overriden by setting register_type to False.
        Further, as with explicit subclassing declarations, the option overtype is
        available, defaulting to False.
        """
        if metacharacters:
            if isinstance(typespec := metacharacters[0], TypeSpec):
                if not issubclass(typespec.archetype, arc):
                    raise TypeError
                metacharacters = list(typespec.metacharacters) + metacharacters[1:]
                kwargs = typespec.metadata | kwargs
        metadata_keys = set(arc.metadata.fields) & set(kwargs)
        metadata = {k: v for k, v in kwargs.items() if k in metadata_keys}
        spec = arc.__typespec_interperter__(*metacharacters, **metadata)
        archetype: Archetype = spec.archetype
        if archetype != arc:
            # This a metamorphism case, we must switch the basetype accordingly
            if basetype is None or not issubclass(basetype, archetype):
                basetype = archetype.basetype
            return archetype.subtype(
                *metacharacters,
                basetype=basetype,
                type_name=type_name,
                register_type=register_type,
                overtype=overtype,
                **kwargs,
            )
        try:
            subtype_ = arc._types[spec]
            if basetype is not None:
                assert issubclass(subtype_, basetype)
            if type_name is not None:
                assert type_name == subtype_.__name__
            return subtype_
        except (KeyError, AssertionError):
            return arc.__subtype__(
                spec,
                type_name=type_name,
                basetype=basetype,
                register_type=register_type,
                overtype=overtype,
                **kwargs,
            )

    def register_trait(arc: type[Arche], metacharacter: str, trait: type[T]):
        """Registers a new trait."""
        # In order to preserve method resolution order, the metacharacter is
        # popped prior to insertion
        try:
            arc._metacharacters.remove(metacharacter)
        except ValueError:
            pass
        try:
            arc._traits.pop(metacharacter)
        except KeyError:
            pass
        arc._metacharacters.append(metacharacter)
        arc._traits[metacharacter] = trait

    def __getitem__(arc, key) -> type[T]:
        """Syntactic help to access types from archetypes with a registration key."""
        return arc.retrieve_type_from_key(key, subtype=True)

    def __getattr__(arc, name: str):
        """Exposes the attributes of Base."""
        if name in {"__kwargs__", "__type_kwargs__"}:
            return getattr(arc.Base, name)
        elif not name.startswith("_"):
            return getattr(arc.Base, name)

    def __setattr__(arc, name: str, value: Any) -> None:
        if not arc._writable:
            msg = f"Cannot set attributes of an Archetype"
            raise AttributeError(msg)
        return super().__setattr__(name, value)

    def __repr__(arc):
        return f"<archetype:{arc.__name__}>"


def _archetype(
    basetype: type[T], *, metatype: Optional[type[type]] = None
) -> Archetype[T]:
    """The archetype decorating function."""
    return Archetype._make_archetype(basetype, metatype=metatype)


def archetype(
    basetype: Optional[type[T]] = None, *, metatype: Optional[type[type]] = None
):
    """Type decorator to declare new archetypes."""
    if metatype is not None:
        if basetype is not None:
            msg = "Improper call to archetype decorator"
            raise TypeError(msg)

        def decorator(basetype: type[T]) -> Archetype[T]:
            return _archetype(basetype, metatype=metatype)

        return decorator
    else:
        return Archetype._make_archetype(basetype)


def metamorph(archetype_: Archetype[T]) -> Archetype[T]:
    """Registers an archetype as a metamorphism for its parent."""
    basetype = archetype_.basetype
    basetype_spec: TypeSpec = getattr(basetype, "typespec")
    parent_archetype: type[Arche] = basetype.archetype
    metamorphisms = parent_archetype.metamorphisms
    metamorphisms[basetype_spec] = archetype_
    return archetype_
