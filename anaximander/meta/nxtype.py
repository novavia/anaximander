from functools import wraps
from typing import Callable, Generic, Iterable, TypeVar, Union, ClassVar

from pydantic import BaseModel


from ..utilities.functions import camel_to_snake, typeproperty
from ..utilities.nxdataclasses import DictLikeDataClass, pydantic_model_class
from ..utilities.nxyaml import yaml_object, yaml
from ..descriptors.fieldtypes import UNDEFINED
from ..descriptors.base import DescriptorRegistry, SettingRegistry
from ..descriptors.typespec import TypeSpec
from .arche import Arche
from .archetype import Archetype
from .nxobject import nxobject, nxtrait

T = TypeVar("T", bound=nxobject, covariant=True)
NotImplementedType = type(NotImplemented)


def trait_init_name(trait: type["nxtype"]):
    """This function renames a trait's __init__ method in a predictable way."""
    return "__" + camel_to_snake(trait.__name__) + "_init__"


def append_trait_inits(init: Callable):
    """Appends traits init methods to a type's regular __init__ method."""

    @wraps(init)
    def __init__(*args, **kwargs):
        result = init(*args, **kwargs)


@yaml_object(yaml)
class nxtype(type, Generic[T]):
    """Base metaclass for concrete Anaximander types."""

    # The metaclass' corresponding archetype, set by Metatype
    __archetype__: ClassVar[Arche]

    # Pydantic model classes for validating instances
    __type_metadata__: type[BaseModel] = pydantic_model_class("TypeMetadataModel")
    __metadata__: type[BaseModel] = pydantic_model_class("MetadataModel")
    __options__: type[BaseModel] = pydantic_model_class("OptionModel")
    # Admissible keyword arguments for subclasses and instances
    __type_kwargs__: set[str] = set()
    __kwargs__: set[str] = set()

    # The descriptor registry
    __descriptors__: DescriptorRegistry
    # The type-level settings registry
    __settings__: SettingRegistry

    # Pydantic model instances for type-level attributes
    _metadata: BaseModel = __metadata__()
    _options: BaseModel = __options__()
    _metacharacters: set[str] = set()

    def __new__(
        mcl,
        name,
        bases,
        namespace,
        register_type: bool = True,
        overtype: bool = False,
        metacharacters: tuple[str] = (),
        **kwargs,
    ):
        """This method underlies type interpretation and creation."""
        # Only one base type is accepted through the declarative interface
        try:
            assert len(bases) == 1
        except AssertionError:
            msg = "Only one base type or archetype is allowed in the inheritance chain."
            raise TypeError(msg)
        # Enforce that the basetype subclasses the metaclass' archetype
        basetype: type[nxobject] = bases[0]
        archetype = mcl.__archetype__
        try:
            assert issubclass(basetype, archetype)
        except AssertionError:
            msg = f"New {mcl.__name__} must subclass {archetype}, cannot use {basetype} as base type."
            raise TypeError(msg)
        # Lastly, if the basetype is archetype.Base, we switch it to the
        # archetype's basetype. This declutters the inheritance chain.
        if basetype is archetype.Base:
            basetype = archetype.basetype

        # Data descriptors (and only data descriptors) are collected from
        # the namespace and turned into a data specifications that is passed as
        # metadata
        data_basetype = archetype.Base if basetype is archetype.basetype else basetype
        data_interpreter = getattr(data_basetype, "__data_interpreter__", None)
        if data_interpreter:
            specifications: dict = data_interpreter(namespace, new_type_name=name)
            for key, value in specifications.items():
                kwargs.setdefault(key, value)

        spec = archetype.__typespec_interperter__(*metacharacters, **kwargs)
        spec_archetype: type[Arche] = spec.archetype
        if spec_archetype != archetype:
            # This a metamorphism case, we must switch the basetype accordingly
            if not issubclass(basetype, spec_archetype):
                basetype = spec_archetype.basetype
        return spec_archetype.__subtype_new__(spec, name, basetype, namespace, **kwargs)

    def __init__(
        cls,
        name,
        bases,
        namespace,
        register_type=True,
        overtype=False,
        metacharacters: tuple[str] = (),
        **kwargs,
    ):
        """Initializes type attributes."""
        # Reset the metacharacter to None
        cls.__metacharacter__ = None
        archetype = cls.archetype
        basetype: type[nxobject] = bases[0]

        # We add the metacharacters
        cls._metacharacters = set(metacharacters) | archetype.basetype.metacharacters
        # We create the type's metadata and options models from settings
        settings = cls.__settings__
        metadata = settings.fetch("metadata", recursive=True)
        cls._metadata = basetype.__type_metadata__(**metadata)
        options = settings.fetch("options", recursive=True)
        cls._options = basetype.__options__(**options)
        init_kwargs = getattr(cls, "__init_kwargs__", lambda c: set())
        cls.__kwargs__: set[str] = init_kwargs()

        # Then we update the instance-level model classes
        type_metadata, options = cls.typespec.metadata, cls._options.dict(by_alias=True)
        cls.__type_metadata__ = archetype.metadata.model_class(
            all_optional=True, objectspec=False, freeze_settings=True, **type_metadata
        )
        cls.__metadata__ = archetype.metadata.model_class(
            **metadata, freeze_settings=True
        )
        cls.__options__ = archetype.options.model_class(
            all_optional=True, freeze_settings=False, **options
        )

        # And we generate methods based on the archetype's metamethods
        for metamethod in archetype.metamethods:
            metamethod.set_attribute(cls)

        # We then compute the __init__ method from the base type and traits
        init_decorator = cls._init_decorator(*cls.traits)
        cls.__original_init__ = cls.__init__ if "__init__" in vars(cls) else None
        cls.__init__ = init_decorator(cls.__init__)

        # Finally we register the new type, provided either metadata or
        # metacharacters are set
        if (metadata or metacharacters) and register_type:
            cls.__register__(overtype=overtype)

    @classmethod
    def _init_decorator(mcl, *traits):
        """Generates a decorator for the __init__ method based on traits."""
        initializers = []
        # Initializers are applied in reverse mro, i.e. the most recently inherited
        # trait's init method runs last
        for t in reversed(traits):
            init_name = trait_init_name(t)
            init = getattr(t, init_name, None)
            if init is not None:
                initializers.append(init)

        def init_decorator(type_init: Callable):
            if not initializers:
                return type_init

            @wraps(type_init)
            def wrapped(self, *args, **kwargs):
                result = type_init(self, *args, **kwargs)
                for init in initializers:
                    init(self)
                return result

            return wrapped

        return init_decorator

    def __register__(cls, *, overtype: bool = False):
        """Registers a class by computing its type specification.

        If overtype is True, the method may overwrite a previous registration
        with the same spec. Otherwise if the spec is already in the registry,
        the class is not registered.
        # TODO: add a warning in that case
        """
        registry = type(cls).__archetype__._types
        # new type registration
        if (spec := cls.typespec) in registry and not overtype:
            # TODO: add warning
            pass
        else:
            registry[spec] = cls

    def __deregister__(cls):
        """Removes class from the metaclass registry.

        This is used by the archetype and trait decorators to remove
        these special types from registries.
        """
        registry = type(cls).__archetype__._types
        spec = cls.typespec
        try:
            assert registry[spec] == cls
        except (KeyError, AssertionError):
            pass
        else:
            del registry[spec]

    @typeproperty
    def archetype(cls) -> Archetype[T]:
        return type(cls).__archetype__

    @typeproperty
    def metadata(cls) -> DictLikeDataClass:
        return cls._metadata.dictlike_dataclass()

    @typeproperty
    def traits(cls) -> tuple["nxtype"]:
        """Returns a tuple of inherited traits, in method resolution order."""
        return tuple(t for t in cls.mro() if isinstance(t, nxtrait))

    @typeproperty
    def metacharacters(cls) -> set[str]:
        """The metacharacters implemented by this type.

        This set may not match traits because only metacharacters that
        are additional to the archetype's basetype are reported here,
        consistent with how type specs are normalized. Second, there may be
        metacharacters that only serve as a routing mechanism for metamorphisms
        and don't implement traits.
        """
        return set(cls._metacharacters)

    @typeproperty
    def typespec(cls) -> TypeSpec:
        """Returns the class' type specification."""
        archetype = type(cls).__archetype__
        return archetype.__typespec_interperter__(*cls.metacharacters, **cls.metadata)

    @typeproperty
    def options(cls) -> DictLikeDataClass:
        return cls._options.dictlike_dataclass()

    @typeproperty
    def unset_typespec_metadata(cls) -> tuple[str]:
        """A tuple of unset typespec metadata fields.

        This tuple being non-empty makes the type abstract.
        """
        schema = type(cls).__archetype__.metadata
        candidates = {
            name: field
            for name, field in schema.fields.items()
            if field.typespec is True and field.default is UNDEFINED
        }
        return tuple(c for c in candidates if getattr(cls, c) is None)

    @typeproperty
    def abstract(cls) -> bool:
        return bool(cls.unset_typespec_metadata)

    @typeproperty
    def anonymous(cls) -> bool:
        """True if cls bears the same name as its archetype."""
        archetype = type(cls).__archetype__
        return cls.__name__ == archetype.__name__

    @typeproperty
    def metacharacter(cls) -> Union[str, NotImplementedType]:
        """A trait's metacharacter, or NotImplemented."""
        if isinstance(cls, nxtrait):
            return getattr(cls, "__metacharacter__")
        return NotImplemented

    def __repr__(cls):
        heading = "nxtrait" if isinstance(cls, nxtrait) else "nxtype"
        if cls.anonymous:
            return repr(cls.typespec).replace("typespec", heading)
        else:
            return f"<{heading}:{cls.__name__}>"

    def __str__(cls):
        heading = "trait" if isinstance(cls, nxtrait) else "nxtype"
        spec_string = str(cls.typespec)
        if cls.anonymous:
            return spec_string.replace("typespec", heading)
        else:
            return spec_string.replace(
                "typespec", f"{heading}\ntype_name: {cls.__name__}"
            )

    @classmethod
    def to_yaml(mcl, representer, node: "nxtype"):
        return node.typespec.to_yaml(representer, node.typespec)


# Sets Arche's metatype to nxtype and nxtype's __archetype__ to Arche
Arche.metatype = nxtype
nxtype.__archetype__ = Arche
