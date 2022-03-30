"""An add-on to dataclasses that adds runtime type enforcement.

The implementation is enabled by Pydantic. Pydantic itself provides a
dataclass alternative for the same functionality but as of this writing the
class' attributes are not detected by Pylance and that makes it harder to
code with.
"""

from collections import ChainMap
from collections.abc import Mapping
import dataclasses as dc
from dataclasses import Field as DataClassField
import types
from typing import Any, Mapping

import pydantic
import pydantic.fields as pydantic_fields


def is_frozen(cls: type) -> bool:
    """True if cls is a frozen dataclass."""
    if dc.is_dataclass(cls):
        return getattr(getattr(cls, "__dataclass_params__"), "frozen", False)
    return False


class DataClassType(type):
    """An abstract metaclass for data classes to facilitate type hinting."""

    def __instancecheck__(self, instance) -> bool:
        return dc.is_dataclass(type(instance))

    def __subclasscheck__(cls, subclass) -> bool:
        return dc.is_dataclass(subclass)


class DataClass(metaclass=DataClassType):
    pass


def field_info(field: DataClassField) -> pydantic_fields.FieldInfo:
    """Extracts a FieldInfo from a dataclass field."""
    if not isinstance(default := field.default, dc._MISSING_TYPE):
        return pydantic.Field(default=default)
    elif not isinstance(default_factory := field.default_factory, dc._MISSING_TYPE):
        return pydantic.Field(default_factory=default_factory)
    else:
        return pydantic.Field()


def dc_field(field: pydantic_fields.FieldInfo) -> DataClassField:
    """Makes a dataclass field from a pydantic field info."""
    if not isinstance(default := field.default, pydantic_fields.UndefinedType):
        return dc.field(default=default)
    elif (default_factory := field.default_factory) is not None:
        return dc.field(default_factory=default_factory)
    else:
        return dc.field(default=None)


def pydantic_model_class(
    name: str,
    *,
    fields: dict[str, pydantic_fields.FieldInfo] = None,
    annotations: dict = None,
    validators: dict = None,
) -> type[pydantic.BaseModel]:
    """Makes a pydantic model class from attributes and optional validators.

    Args:
        fields: a mapping of attribute names to FieldInfo objects
        annotations: a mapping of attribute names to type hints
        validators: an optional mapping of method name to validator methods,
            per pydantic specifications.

    The generated model class also stores a dictlike, frozen dataclass
    that can be used to export models as model / dictionary hybrids.
    """
    if fields is None:
        if annotations is None:
            fields = {}
        else:
            fields = {k: pydantic.Field() for k in annotations}
    export_fields = {(v.alias or k): dc_field(v) for k, v in fields.items()}
    if annotations is None:
        annotations = {}
    export_annotations = {
        (v.alias or k): annotations.get(k, Any) for k, v in fields.items()
    }
    if validators is None:
        validators = {}
    export_namespace = export_fields
    export_namespace["__annotations__"] = export_annotations
    export_exec_body = lambda ns: ns.update(export_namespace)
    export_cls = dictlike(
        dc.dataclass(types.new_class(name, exec_body=export_exec_body), frozen=True)
    )

    namespace = fields | validators
    namespace["__annotations__"] = annotations
    namespace["__dictlike_dataclass__"] = export_cls

    def dictlike_dataclass(self):
        """Returns a copy of self as a plain dataclass with dict-like access and iteration."""
        return self.__dictlike_dataclass__(**self.dict(by_alias=True))

    namespace["dictlike_dataclass"] = dictlike_dataclass

    # We also define a __copy__ method on model classes
    def __copy__(self):
        return type(self)(**self.dict())

    namespace["__copy__"] = __copy__
    exec_body = lambda ns: ns.update(namespace)
    model_cls = types.new_class(name, (pydantic.BaseModel,), exec_body=exec_body)
    return model_cls


class Validated:
    """A dataclass mix-in that parses inputs into declared type hints."""

    def __init_subclass__(cls) -> None:
        cls.__set_pydantic_model_class__()

    @classmethod
    def __set_pydantic_model_class__(cls) -> None:
        """Generates a Pydantic model from annotations."""
        # To avoid infinite recursion due to the new_class call, we set
        # and detect a flag
        if getattr(cls, "__i_am_a_deep_fake__", None) is True:
            return
        name = cls.__name__
        bases = cls.__bases__
        annotations = ChainMap(*[getattr(c, "__annotations__", {}) for c in cls.mro()])
        namespace = cls.__dict__.copy()
        namespace["__i_am_a_deep_fake__"] = True
        exec_body = lambda ns: ns.update(namespace)
        alt_cls = types.new_class(name, bases, exec_body=exec_body)
        fields = dc.fields(dc.dataclass(alt_cls, frozen=is_frozen(alt_cls)))
        field_infos = {f.name: field_info(f) for f in fields}
        cls.__pydantic_model__ = pydantic_model_class(
            name=cls.__name__, fields=field_infos, annotations=annotations
        )

    def __post_init__(self):
        self.__pydantic_model__(**dc.asdict(self))


def validate_data(obj: Validated, *, return_model=False):
    """Validates a dataclass instance that inherits from Validated."""
    model_class = obj.__pydantic_model__
    data = {f.name: getattr(obj, f.name) for f in dc.fields(obj)}
    model = model_class(**data)
    if return_model:
        return model


class DataClassMappingMixin(Mapping):
    """A mixin class that makes data classes behave as frozen dictionaries."""

    def __init_subclass__(cls, **kwargs) -> None:
        if not is_frozen(cls):
            msg = "Only frozen dataclasses can implement DataClassMappingMixing"
            raise TypeError(msg)
        return super().__init_subclass__(**kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(dc.asdict(self))

    def __len__(self):
        return len(dc.asdict(self))


class DictLikeDataClassType(DataClassType):
    """An abstract metaclass for dict-like data classes to facilitate type hinting."""

    def __subclasscheck__(cls, subclass) -> bool:
        return dc.is_dataclass(subclass) and issubclass(cls, DataClassMappingMixin)


class DictLikeDataClass(metaclass=DictLikeDataClassType):
    pass


def dictlike(cls: type) -> DictLikeDataClass:
    """A dataclass decorator that adds dict-like access and iteration."""
    return types.new_class(cls.__name__, (DataClassMappingMixin, cls))
