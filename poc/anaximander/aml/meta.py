"""This module defines base types and metaclasses for the Anaximander Modeling Language (AML)."""

import datetime
from abc import ABC, ABCMeta
from collections.abc import Collection
from typing import Any, Callable, ClassVar, Protocol, TypeVar

import attrs
from pydantic import BaseModel


@attrs.define
class Metadescriptor(ABC):
    """Base class for all metadescriptors.

    Metadescriptors define attributes of primitive data types and models.
    They are called metadescriptors because they are not proper descriptors, but rather
    declarations that are used to generate descriptors in compiled types.
    """

    __reserved_names__: ClassVar[Collection[str]] = set()
    name: str = attrs.field(init=False)

    def __init_subclass__(cls):
        super().__init_subclass__()
        parent = cls.mro()[1]
        reserved_names: set[str] = getattr(parent, "__reserved_names__", set())
        if "__reserved_names__" in vars(cls):
            cls.__reserved_names__ = reserved_names | set(cls.__reserved_names__)
        else:
            cls.__reserved_names__ = reserved_names

    def __set_name__(self, owner: type, name: str):
        if name in self.__reserved_names__:
            mdtype = self.__class__.__name__
            msg = f"Cannot use reserved name {name} for Metadescriptor or type {mdtype}."
            raise ValueError(msg)
        self.name = name


class Prototype(ABCMeta):
    """Metaclass for model and data declarative types."""

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.__compilations__: dict[str, dict] = {}

    def metadescriptors(cls, *types: type[Metadescriptor], inherited: bool = True):
        """Returns a dictionary of metadescriptors of the supplied types.

        If inherited is set to True, metadescriptors declared in parent models are included.
        Otherwise, only the metadescriptors directly declared by cls are returned.
        """
        if not types:
            types = (Metadescriptor,)
        cls_metadescriptors = {k: v for k, v in cls.__dict__.items() if isinstance(v, types)}
        if inherited:
            parent = cls.mro()[1]
            if isinstance(parent, Prototype):
                metadescriptors = (
                    parent.metadescriptors(*types, inherited=True) | cls_metadescriptors
                )
            else:
                metadescriptors = cls_metadescriptors
        else:
            metadescriptors = cls_metadescriptors
        return metadescriptors


class DataABC(ABC):
    """Abstract base class for data types."""

    pass


class ModelABC(ABC):
    """Abstract base class for model types."""

    pass


class DataObjectABC(ABC):
    """Abstract base class for data objects.

    DataObjects are concrete instances of model-space representations,
    which include primitive data types, models, and aggregate structures
    built from these.
    """

    pass


type data = (
    DataABC
    | bool
    | int
    | float
    | str
    | bytes
    | datetime.date
    | datetime.datetime
    | datetime.time
    | datetime.timedelta
)


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


type model = ModelABC | BaseModel | Dataclass

type prototype = type[DataABC] | type[model]

type dataobject = DataObjectABC | data | model | Collection[dataobject]

P = TypeVar("P", bound=prototype)


def compile(*compilers: str, **kwargs) -> Callable[[P], P]:
    """A class decorator factory that flags a model for compilations."""

    def decorator(cls: P) -> P:
        if not hasattr(cls, "__compilations__"):
            cls.__compilations__: dict[str, dict] = {}  # type: ignore
        for handle in compilers:
            cls.__compilations__[handle] = dict(kwargs)  # type: ignore
        return cls

    return decorator


@attrs.define
class TypedMetadescriptor(Metadescriptor):
    hint: Any = attrs.field(init=False)


@attrs.define
class DataobjectMetadescriptor(TypedMetadescriptor):
    hint: type[dataobject] = attrs.field(init=False)
