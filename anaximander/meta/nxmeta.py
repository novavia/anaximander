from typing import Optional, Any

from ..descriptors.base import SettingRegistry
from .arche import Arche


class nxmeta(type):
    """The metaclass for Anaximander metatypes."""

    def __new__(
        nxmeta, name, bases, namespace, archetype: Optional[type[Arche]] = None
    ):
        # This is necessary for creating custom metaclasses
        parent_ns = getattr(bases[0], "__parent_namespace__", dict())
        if archetype is None:
            boilerplate = {
                "__module__",
                "__qualname__",
                "__doc__",
                "__parent_namespace__",
            }
            functional_ns = {k: v for k, v in namespace.items() if k not in boilerplate}
            parent_ns.update(functional_ns)
            namespace["__parent_namespace__"] = parent_ns
        else:
            namespace.update(parent_ns)
            namespace["__parent_namespace__"] = dict()
        return super().__new__(nxmeta, name, bases, namespace)

    def __init__(mcl, name, bases, namespace, archetype: Optional[type[Arche]] = None):
        """Initializes a new metaclass."""
        if archetype is None:
            return

        mcl.__archetype__ = archetype
        basetype = archetype.basetype
        # We set type-level model classes
        settings: SettingRegistry = getattr(archetype.basetype, "__settings__")
        metadata = settings.fetch("metadata", recursive=True)
        options = settings.fetch("options", recursive=True)
        mcl.__type_metadata__ = archetype.metadata.model_class(
            all_optional=True, objectspec=False, freeze_settings=True, **metadata
        )
        mcl.__metadata__ = archetype.metadata.model_class(
            **metadata, freeze_settings=True
        )
        mcl.__options__ = archetype.options.model_class(
            all_optional=True, freeze_settings=False, **options
        )

        # Then we set type attribute descriptors
        new_fields = archetype.descriptors.fetch("metadata", "options")
        for field in new_fields.values():
            field.set_attribute(mcl)

        # As well as metaproperties
        metaproperties = archetype.metaproperties
        for metaprop in metaproperties:
            metaprop.set_attribute(mcl)

        # And metamethods
        metamethods = archetype.metamethods
        for metameth in metamethods:
            setattr(mcl, metameth.name, metameth)

    @property
    def __unset_typespec_metadata__(mcl) -> tuple[str]:
        """Returns an enumeration of type-level metadata attributes that are settable"""
        candidates = [
            name
            for name, field in mcl.__archetype__.metadata.fields.items()
            if field.typespec is True
        ]
        return tuple(
            f.name
            for f in mcl.__type_metadata__.__fields__.values()
            if not f.field_info.const and f.name in candidates
        )
