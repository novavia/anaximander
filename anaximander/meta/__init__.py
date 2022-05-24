__all__ = [
    "Object",
    "archetype",
    "nxmeta",
    "nxtype",
    "nxobject",
    "nxtrait",
    "trait",
    "metamorph",
]


from ..utilities.nxdataclasses import DictLikeDataClass

from ..descriptors.typespec import TypeSpec
from ..descriptors.base import SettingRegistry

from .archetype import archetype, Archetype, metamorph
from .nxmeta import nxmeta
from .nxobject import nxobject, nxtrait
from .nxtype import nxtype
from .trait import trait


@archetype
class Object(nxobject, metaclass=nxtype):
    """The base class for application objects."""

    def __new__(cls: nxtype, *args, _run_interpreter=True, **kwargs):
        """The _run_interpreter flag avoids infinite recursion in fetching subtypes."""
        if not _run_interpreter:
            if not cls.abstract:
                return super().__new__(cls)
            else:
                msg = f"Cannot instantiate abstract type {cls}"
                raise TypeError(msg)

        # Field settings are collected from the kwargs
        settings_registry = SettingRegistry(cls.__descriptors__)
        settings_registry.collect(kwargs)
        metadata = settings_registry.fetch("metadata")
        # If there is no metadata, then no type manipulation is necessary
        # and we can make the instance
        if not metadata and not cls.abstract:
            return super().__new__(cls)

        # Otherwise we merge the metadata with the type's metadata
        # and we validate the merge
        metadata = dict(cls.metadata) | metadata
        cls.__metadata__(**metadata)

        # Then we resolve the type
        subtype: nxtype = cls.archetype.subtype(basetype=cls, **metadata)
        object_instance = subtype.__new__(
            subtype, *args, _run_interpreter=False, **kwargs
        )
        return object_instance

    @property
    def archetype(self) -> Archetype:
        return type(self).archetype

    @property
    def metacharacters(self) -> set[str]:
        return type(self).metacharacters

    @property
    def traits(self) -> tuple[nxtype]:
        return type(self).metacharacters

    @property
    def metadata(self) -> DictLikeDataClass:
        return self._metadata.dictlike_dataclass()

    @property
    def options(self) -> DictLikeDataClass:
        return self._options.dictlike_dataclass()

    @property
    def typespec(self) -> TypeSpec:
        return type(self).typespec

    def __init__(self, **kwargs):
        # We validate the keyword arguments
        unknown_kwargs = set(kwargs) - type(self).__kwargs__
        if unknown_kwargs:
            msg = f"Unknown keyword arguments {unknown_kwargs}"
            raise TypeError(msg)
        settings_registry = SettingRegistry(self.__descriptors__)
        settings_registry.collect(kwargs)
        metadata = settings_registry.fetch("metadata")
        self._metadata = self.__metadata__(**metadata)
        options = settings_registry.fetch("options")
        self._options = self.__options__(**options)

    def __eq__(self, other):
        if not (same_type := (type(self) == type(other))):
            return False
        return self.metadata == other.metadata

    def __str__(self):
        return repr(self)
