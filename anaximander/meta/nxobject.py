from typing import ClassVar, Optional

from pydantic import BaseModel

from ..utilities.nxdataclasses import DictLikeDataClass, pydantic_model_class

from ..descriptors.typespec import TypeSpec, typespec
from ..descriptors.base import Descriptor, DescriptorRegistry, SettingRegistry
from .arche import Arche

NotImplementedType = type(NotImplemented)


class nxobject(object):
    """Base class for Anaximander concrete types."""

    # Class variables, set anew by metaclasses
    # The descriptor registry, which collects descriptors declared in subclasses
    __descriptors__: ClassVar[DescriptorRegistry]
    # The type-level settings registry
    __settings__: ClassVar[SettingRegistry]
    # If not None, the type is a trait
    __metacharacter__: ClassVar[Optional[str]] = None

    # Pydantic model classes for type and instance attributes
    __type_metadata__: ClassVar[type[BaseModel]] = pydantic_model_class(
        "TypeMetadataModel"
    )
    __metadata__: ClassVar[type[BaseModel]] = pydantic_model_class("MetadataModel")
    __options__: ClassVar[type[BaseModel]] = pydantic_model_class("OptionsModel")

    # Add attributes for compatibility with the nxtype interface
    # These are all implemented as type properties in nxtype
    archetype: ClassVar[type[Arche]] = Arche
    metadata: ClassVar[DictLikeDataClass] = dict()
    traits: ClassVar[tuple[type]] = tuple()
    metacharacters: ClassVar[set[str]] = set()
    _metacharacters: ClassVar[set[str]] = set()
    typespec: ClassVar[TypeSpec] = typespec(Arche)
    options: ClassVar[DictLikeDataClass] = dict()
    unset_typespec_metadata: ClassVar[tuple[str]] = tuple()
    abstract: ClassVar[bool] = True
    anonymous: ClassVar[bool] = False
    metacharacter: ClassVar[NotImplementedType] = NotImplemented

    # The instance-level settings registry
    _settings: SettingRegistry

    # Pydantic model instances
    _metadata: BaseModel = __metadata__()
    _options: BaseModel = __options__()

    def __repr__(self):
        return repr(type(self)).replace("nxtype", "nxobject")

    def __str__(self):
        return str(type(self)).replace("nxtype", "nxobject")

    @classmethod
    def __data_interperter__(cls, namespace, *, new_type_name: str):
        """Special method that interprets namespace declarations to resolve data specifications."""
        return {}

    @classmethod
    def __init_kwargs__(cls) -> set[str]:
        """Makes admissible keyword arguments for instances."""
        return getattr(cls, "__kwargs__", set())


# Sets Arche's base type to nxobject
Arche.basetype = nxobject
Arche.register(nxobject)
# Creates registries for metadescriptors and settings
Descriptor.set_up_registry(nxobject)
nxobject.__settings__ = SettingRegistry(nxobject.__descriptors__)


class TraitMeta(type):
    """Abstract base metaclass for traits."""

    __registry__: ClassVar[set[type[nxobject]]] = set()

    @classmethod
    def register(mcl, type_: type[nxobject]):
        mcl.__registry__.add(type_)

    def __instancecheck__(cls, type_: type) -> bool:
        return type_ in cls.__registry__


# The nxtrait abstract metaclass registers subclasses of nxobject that are traits
class nxtrait(type, metaclass=TraitMeta):
    """Abstract metaclass for traits."""

    __metacharacter__: str
