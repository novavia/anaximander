"""This module defines base types and metaclasses for the Anaximander Modeling Language (AML)."""

import datetime
from abc import ABC, ABCMeta
from collections.abc import Collection
from enum import Enum
from typing import Any, Callable, ClassVar, TypeVar, get_type_hints

import attrs
from annotationlib import Format, get_annotations

# from pydantic import BaseModel   # pydantic not yet compatible with python 3.14


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


M = TypeVar("M", bound=Metadescriptor)


class Prototype(ABCMeta):
    """Metaclass for model and data declarative types."""

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.__compilations__: dict[str, dict] = {}

    def metadescriptors(cls, *types: type[M], inherited: bool = True) -> dict[str, M]:
        """Returns a dictionary of metadescriptors of the supplied types.

        If inherited is set to True, metadescriptors declared in parent models are included.
        Otherwise, only the metadescriptors directly declared by cls are returned.
        """
        if not types:
            types = (Metadescriptor,)  # type: ignore
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

    def __set_type_annotations__(cls):
        """Sets type annotations on typed metadescriptors."""
        annotations = get_annotations(cls, format=Format.STRING)
        superhints = get_type_hints(cls)  # This includes super classes
        metadescriptors = cls.metadescriptors(TypedMetadescriptor, inherited=False)
        for name, metadescriptor in metadescriptors.items():
            metadescriptor.annotation = annotations.get(name, "")
            metadescriptor.hint = superhints.get(name, None)


class DataABC(ABC):
    """Abstract base class for data types."""

    __primitives__ = (
        bool,
        int,
        float,
        str,
        bytes,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
    )

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if issubclass(subclass, Enum):
            return all(isinstance(m._value_, cls) for m in subclass.__members__.values())
        return super().__subclasshook__(subclass)


for primitive in DataABC.__primitives__:
    DataABC.register(primitive)

data = DataABC  # literal alias for enhanced readability


class ModelABC(ABC):
    """Abstract base class for model types."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        # case dataclass
        if hasattr(subclass, "__dataclass_fields__"):
            return True
        return super().__subclasshook__(subclass)


# ModelABC.register(BaseModel)  # pydantic not yet compatible with python 3.14

model = ModelABC  # literal alias for enhanced readability


class DataObjectABC(ABC):
    """Abstract base class for data objects.

    DataObjects are concrete instances of model-space representations,
    which include primitive data types, models, and aggregate structures
    built from these.
    """

    pass


DataObjectABC.register(DataABC)
DataObjectABC.register(ModelABC)


prototype = type[DataABC] | type[ModelABC]

type dataobject = DataObjectABC | DataABC | ModelABC | Collection[dataobject]


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
    annotation: str = attrs.field(init=False)  # Literal type annotation as a string
    hint: Any = attrs.field(init=False)  # Evaluated type annotation


@attrs.define
class DataobjectMetadescriptor(TypedMetadescriptor):
    hint: dataobject = attrs.field(init=False)
