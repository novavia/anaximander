"""This module defines base types and metaclasses for the Anaximander Modeling Language (AML)."""

import ast
import datetime
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Collection
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

import attrs
from annotationlib import Format, get_annotations

from ..utils.funcs import type_name_to_collection_name

type Assignment = ast.Assign | ast.AnnAssign

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
    __ast__: Assignment = attrs.field(init=False)

    def __init_subclass__(cls):
        super().__init_subclass__()
        parent = cls.mro()[1]
        reserved_names: set[str] = getattr(parent, "__reserved_names__", set())
        if "__reserved_names__" in vars(cls):
            cls.__reserved_names__ = reserved_names | set(cls.__reserved_names__)
        else:
            cls.__reserved_names__ = reserved_names

    def __set_name__(self, owner: "Prototype", name: str):
        if name in self.__reserved_names__:
            mdtype = self.__class__.__name__
            msg = f"Cannot use reserved name {name} for Metadescriptor or type {mdtype}."
            raise ValueError(msg)
        self.name = name


M = TypeVar("M", bound=Metadescriptor)


class Prototype(ABCMeta):
    """Metaclass for model and data declarative types."""
    __compilations__: dict[str, dict]  # Holds compilation target handles and parameters
    __ast__: ast.ClassDef  # Holds the model's parsed abstract syntax tree

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.__compilations__: dict[str, dict] = {}

    def metadescriptors(cls, *types: type[M], inherited: bool = True) -> dict[str, M]:
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

    def __validate_bases__(cls):
        """The first base must be a Prototype and there can be only one."""
        bases = cls.__bases__
        if not isinstance(bases[0], Prototype):
            msg = "Prototypes cannot be used as mixin classes."
            raise TypeError(msg)
        extra_parent_prototypes = [b for b in bases[1:] if isinstance(b, Prototype)]
        if extra_parent_prototypes:
            msg = "Prototypes do not support multiple inheritance."
            raise TypeError(msg)

    def __set_type_annotations__(cls):
        """Sets type annotations on typed metadescriptors."""
        annotation_values = get_annotations(cls, format=Format.VALUE)
        annotation_strings = get_annotations(cls, format=Format.STRING)
        superhints = get_type_hints(cls)  # This includes super classes
        metadescriptors = cls.metadescriptors(TypedMetadescriptor, inherited=False)
        for name, metadescriptor in metadescriptors.items():
            annotation_value = annotation_values.get(name, "")
            annotation_string = annotation_strings.get(name, "")
            if isinstance(annotation_value, str):
                annotation = f'"{annotation_value}"'
            else:
                annotation = annotation_string
            hint = superhints.get(name, None)
            metadescriptor.__set_type__(cls, annotation, hint)

    @property
    def collection_name(cls):
        """A collection name using camel case and pluralization.

        This can be customized by passing metadata (#TODO).
        """
        return type_name_to_collection_name(cls.__name__)


class DataABC(ABC):
    """Abstract base class for data types."""

    __primitives__ = (
        bool,
        bytes,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
        Decimal,
        float,
        int,
        str,
        UUID,
    )

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if issubclass(subclass, Enum):
            return all(isinstance(m._value_, cls) for m in subclass.__members__.values())
        return super().__subclasshook__(subclass)

    @classmethod
    def primitive(cls, datatype: type) -> type:
        """Returns the primitive python type for a supplied data type."""
        if not isinstance(datatype, type):
            msg = f"Non-type object {datatype} supplied to primitive method."
            raise TypeError(msg)
        if datatype in cls.__primitives__:
            return datatype
        if primitive := getattr(datatype, "__primitive__", None):
            return primitive
        msg = f"Cannot extract primitive type from {datatype}"
        raise TypeError(msg)


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

    __collections__ = (list, tuple, dict, set)

    @classmethod
    def __is_subtype__(cls, hint: Any, *super: type):
        """Runtime check on supplied type hints.

        The optional super types restrict the subclassing test to those instead of cls.
        """
        super = super or (cls,)
        if isinstance(hint, type):
            if issubclass(hint, Enum):
                values = [m._value_ for m in hint.__members__.values()]
                vtypes = {type(v) for v in values}
                return all(cls.__is_subtype__(t, *super) for t in vtypes)
            elif issubclass(hint, dict):
                # This case handles TypedDict
                typed_dict_attrs = {"__required_keys__", "__optional_keys__"}
                try:
                    assert typed_dict_attrs < set(vars(hint))
                except (AttributeError, AssertionError):
                    pass
                else:
                    member_hints = get_type_hints(hint)
                    return all(cls.__is_subtype__(t, *super) for t in member_hints.values())
            return issubclass(hint, super)
        origin = get_origin(hint)
        args = get_args(hint)
        if origin in cls.__collections__:
            return all(cls.__is_subtype__(arg, *super) for arg in args)
        return False


DataObjectABC.register(DataABC)
DataObjectABC.register(ModelABC)


prototype = type[DataABC] | type[ModelABC]

type dataobject = DataObjectABC | Collection[dataobject]


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
    __ast__: ast.AnnAssign = attrs.field(init=False)

    @abstractmethod
    def __validate_hint__(self, owner: Prototype, hint: Any) -> bool:
        return True

    def __set_type__(self, owner: Prototype, annotation: str, hint: Any):
        """Sets the type by supplying annotation (string) and hint (expects a type)."""
        self.annotation = annotation
        if not self.__validate_hint__(owner, hint):
            descriptor = self.name
            owner_name = owner.__name__
            msg = (
                f"Incompatible annotation {annotation} supplied to {descriptor} descriptor "
                + f"of {owner_name}."
            )
            raise TypeError(msg)
        self.hint = hint


@attrs.define
class DataObjectMetadescriptor(TypedMetadescriptor):
    hint: type[dataobject] = attrs.field(init=False)

    def __validate_hint__(self, owner: Prototype, hint: type[dataobject]) -> bool:
        return DataObjectABC.__is_subtype__(hint)
