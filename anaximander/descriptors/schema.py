from collections import defaultdict
from collections.abc import Mapping, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
import dataclasses as dc
from frozendict import frozendict
from itertools import chain
from multiprocessing.sharedctypes import Value
from types import new_class
from typing import Any, Callable, Optional, TypeVar, Generic, Union, ClassVar

import pandas as pd
import pydantic

from ..utilities.functions import auto_label, camel_to_snake, freeze
from ..utilities.nxdataclasses import (
    Validated,
    validate_data,
    pydantic_model_class,
    DataClass,
)
from ..utilities.nxyaml import yaml_object, yaml
from .fieldtypes import UNDEFINED
from .datatypes import Hint, nxmodel, TypedDictMeta, pyhint
from .base import Descriptor, DescriptorRegistry, DataDescriptor, DataModelSpecifier
from .fields import Field, DataField, MetadataField, OptionField
from .dataspec import DataSpec

BasicModelParser = Callable[[dict[str, Any]], dict[str, Any]]
BasicModelValidator = Callable[[dict[str, Any]], bool]
BasicModelFilter = Union[BasicModelParser, BasicModelValidator]
DataModelParser = Callable[[type, dict[str, Any]], dict[str, Any]]
DataModelValidator = Callable[[type, dict[str, Any]], bool]
DataModelFilter = Union[DataModelParser, DataModelValidator]

T = TypeVar("T", bound=Field)


class SchemaBase(Generic[T]):
    pass


@yaml_object(yaml)
@dataclass
class Schema(Validated, SchemaBase[T], DataModelSpecifier):
    yaml_tag: ClassVar[str] = "!schema"

    name: str = dc.field(default="Schema")
    fields: dict[str, T] = dc.field(default_factory=dict)
    input_parser: Optional[BasicModelParser] = dc.field(default=None, repr=False)
    validators: list[BasicModelValidator] = dc.field(default_factory=list, repr=False)
    config: dict = dc.field(default_factory=dict, repr=False)
    _hash: Optional[int] = dc.field(default=None, repr=False)

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        fields: Optional[Iterable[T]] = None,
        parser: Optional[BasicModelParser] = None,
        validators: Optional[Iterable[BasicModelFilter]] = None,
        **config,
    ):
        """Creates a schema from a name and a set of fields."""
        self.name = name or type(self).__name__
        if fields is None:
            self.fields = dict()
        else:
            fields = list(fields)
            self.fields = {f.name: f for f in fields}
            if len(self.fields) < len(fields):
                msg = "Cannot supply duplicated field names to a Schema"
                raise ValueError(msg)
        self.input_parser = parser
        self.validators = list(validators) if validators is not None else []
        self.config = config
        self.__post_init__()
        validate_data(self)
        self._hash = self._hash_function()

    def __post_init__(self):
        pass

    def _hash_function(self):
        return hash(freeze(dc.asdict(self)))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        try:
            return self._hash == getattr(other, "_hash")
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self._hash != getattr(other, "_hash")
        except AttributeError:
            return False

    def __setattr__(self, name: str, value: Any) -> None:
        if self._hash is not None:
            msg = "Cannot set attributes of Schema instances"
            raise AttributeError(msg)
        return super().__setattr__(name, value)

    @contextmanager
    def modify(self):
        hash_ = self._hash
        object.__setattr__(self, "_hash", None)
        yield
        if hash_ is None:
            return
        self._hash = self._hash_function()

    def extend(
        self,
        *,
        name: Optional[str] = None,
        fields: Optional[Iterable[T]] = None,
        parser: Optional[BasicModelParser] = None,
        validators: Optional[Iterable[BasicModelFilter]] = None,
        **config,
    ):
        """Returns a new instance with optional attribute modifications"""
        name = name or self.name
        if fields is not None:
            fields = {f.name: f for f in fields}
            fields = list((self.fields | fields).values())
        parser = parser or self.input_parser
        if validators is None:
            validators = self.validators
        else:
            validators = self.validators + list(validators)
        config = self.config | config
        return type(self)(
            name=name, fields=fields, parser=parser, validators=validators, **config
        )

    def __le__(self, other: "Schema"):
        try:
            assert self.fields.items() <= other.fields.items()
            self_type, other_type = type(self), type(other)
            assert issubclass(self_type, other_type) or issubclass(
                other_type, self_type
            )
            assert self.input_parser == other.input_parser or self.input_parser is None
            assert all(v in other.validators for v in self.validators)
            assert self.config.items() <= other.config.items()
        except AssertionError:
            return False
        else:
            return True

    def __lt__(self, other: "Schema"):
        return self <= other and self != other

    def __gt__(self, other: "Schema"):
        return self >= other and self != other

    def __ge__(self, other: "Schema"):
        try:
            assert self.fields.items() >= other.fields.items()
            self_type, other_type = type(self), type(other)
            assert issubclass(self_type, other_type) or issubclass(
                other_type, self_type
            )
            assert self.input_parser == other.input_parser or other.input_parser is None
            assert all(v in self.validators for v in other.validators)
            assert self.config.items() >= other.config.items()
        except AssertionError:
            return False
        else:
            return True

    def __and__(self, other: "Schema") -> Optional["Schema"]:
        try:
            self_type, other_type = type(self), type(other)
            if issubclass(self_type, other_type):
                type_ = other_type
            elif issubclass(other_type, self_type):
                type_ = self_type
            else:
                raise AssertionError
            if self.name == other.name:
                name = self.name
            else:
                name = self.name + "And" + other.name
            fields = [
                f for f in self.fields.values() if f in set(other.fields.values())
            ]
            if None in (self.input_parser, other.input_parser):
                input_parser = None
            else:
                assert self.input_parser == other.input_parser
                input_parser = self.input_parser
            validators = set(self.validators) & set(other.validators)
            config = dict(self.config.items() & other.config.items())
        except AssertionError:
            return NotImplemented
        return type_(
            name=name,
            fields=fields,
            parser=input_parser,
            validators=validators,
            **config,
        )

    def __or__(self, other: "Schema") -> Optional["Schema"]:
        try:
            self_type, other_type = type(self), type(other)
            if issubclass(self_type, other_type):
                type_ = self_type
            elif issubclass(other_type, self_type):
                type_ = other_type
            else:
                raise AssertionError
            if self.name == other.name:
                name = self.name
            else:
                name = self.name + "Or" + other.name
            fields = list(self.fields.values()) + list(
                f for f in other.fields.values() if f not in set(self.fields.values())
            )
            if None in (self.input_parser, other.input_parser):
                input_parser = self.input_parser or other.input_parser
            else:
                assert self.input_parser == other.input_parser
                input_parser = self.input_parser
            validators = set(self.validators) | set(other.validators)
            config = dict(self.config.items() | other.config.items())
        except AssertionError:
            return NotImplemented
        return type_(
            name=name,
            fields=fields,
            parser=input_parser,
            validators=validators,
            **config,
        )

    def __sub__(self, other: "Schema") -> "Schema":
        try:
            self_type, other_type = type(self), type(other)
            assert issubclass(self_type, other_type) or issubclass(
                other_type, self_type
            )
            type_ = self_type
            name = self.name + "Minus" + other.name
            fields = [
                f for f in self.fields.values() if f not in set(other.fields.values())
            ]
            if self.input_parser is not None or self.validators or self.config:
                return NotImplemented
            else:
                input_parser = None
                validators = []
                config = dict()
        except AssertionError:
            return NotImplemented
        return type_(
            name=name,
            fields=fields,
            parser=input_parser,
            validators=validators,
            **config,
        )

    def __add__(self, other: "Schema") -> "Schema":
        try:
            self_type, other_type = type(self), type(other)
            assert issubclass(self_type, other_type) or issubclass(
                other_type, self_type
            )
            type_ = self_type
            name = self.name + "Plus" + other.name
            fields = list(self.fields.values()) + list(
                f for f in other.fields.values() if f not in set(self.fields.values())
            )
            if self.input_parser != other.input_parser:
                if self.input_parser is None:
                    input_parser = other.input_parser
                elif other.input_parser is None:
                    input_parser = self.input_parser
                else:
                    return NotImplemented
            else:
                input_parser = self.input_parser
            validators = set(self.validators) | set(other.validators)
            config = self.config | other.config
        except AssertionError:
            return NotImplemented
        return type_(
            name=name,
            fields=fields,
            parser=input_parser,
            validators=validators,
            **config,
        )

    @property
    def dispensable(self) -> set[str]:
        """Returns the set of dispensable fields."""
        return {name for name, field in self.fields.items() if field.dispensable}

    @classmethod
    def make_pydantic_validator(
        cls_,
        method: BasicModelFilter,
        *,
        pre: bool = False,
        namespace: Optional[type] = None,
        method_name: Optional[str] = None,
        docstring: Optional[str] = None,
    ):
        """Returns a method wrapped by pydantic's validator.

        :param method: the validation method, carrying the signature of
            a classmethod validating a single argument.
        :param pre: if True, the method outputs a Pydantic 'pre' validator,
            i.e. a parser.
        :param namespace: an optional type that is supplied to the validation
            method for parameter lookup. This is featured here for compatibility
            with the DataSchema.make_pydantic_validator method.
        :param method_name: an optional method name that overrides the
            name of the supplied method. This may be necessary to avoid
            namespace conflicts in the generated Pydantic model.
        :param docstring: an optional docstring attached to the validation
            method.
        """

        if pre is True:
            # Parser use case
            def wrapped(cls, values: dict[str, Any]) -> dict[str, Any]:
                return method(values)

        else:
            # Validation use case
            def wrapped(cls, values: dict[str, Any]) -> dict[str, Any]:
                assert method(values), method.__doc__
                return values

        wrapped.__module__ = method.__module__
        wrapped.__name__ = method_name or method.__name__
        wrapped.__qualname__ = method.__qualname__
        if docstring is not None:
            wrapped.__doc__ = docstring
        else:
            wrapped.__doc__ = method.__doc__
        wrapped.__annotations__ = method.__annotations__

        validator = pydantic.root_validator(pre=pre, allow_reuse=True)(wrapped)
        return validator

    def _normalize_annotations(
        self, annotations: Mapping[str, Hint]
    ) -> Mapping[str, Hint]:
        return annotations

    def _to_pydantic(
        self,
        namespace: Optional[type] = None,
        all_optional: bool = False,
        freeze_settings: bool = True,
        filter: Optional[Callable] = None,
        parse: bool = True,
        validate: bool = True,
        **settings,
    ) -> type[pydantic.BaseModel]:
        """Primitive method for generating Pydantic model classes."""
        field_infos = {}
        annotations = {}
        validators = {}
        if parse:
            if self.input_parser is not None:
                parser = self.make_pydantic_validator(
                    self.input_parser, pre=True, namespace=namespace
                )
                validators[parser.__name__] = field_parser
        if validate:
            for v in self.validators:
                validator = self.make_pydantic_validator(v, namespace=namespace)
                validators[validator.__name__] = validator
        if filter is not None:
            fields = {k: v for k, v in self.fields.items() if filter(v)}
        else:
            fields = self.fields
        for fname, field in fields.items():
            # This checks whether the field is a compound type, i.e. a Data
            # subtype
            dataspec = getattr(field, "base_spec", None)

            setting = settings.get(fname, UNDEFINED)
            if dataspec is not None and isinstance(setting, field.type_):
                setting = setting.data
            display_name = fname
            if field.alias:
                fname = field.alias
            optional = all_optional or field.nullable
            field_info = field.make_pydantic_field_info(
                setting,
                validators=validate,
                optional=optional,
                freeze_setting=freeze_settings,
            )
            if (
                basetype := getattr(field_info, "extra", {}).get("nx_basetype")
            ) is not None:
                v_name = (
                    display_name + "_is_subtype_of_" + camel_to_snake(basetype.__name__)
                )
                validators[v_name] = field._pydantic_subtype_validator(fname, basetype)
            if (
                basespec := getattr(field_info, "extra", {}).get("nx_basespec")
            ) is not None:
                v_name = (
                    display_name + "_is_submodel_of_" + camel_to_snake(basespec.name)
                )
                validators[v_name] = field._pydantic_submodel_validator(fname, basespec)
            field_infos[fname] = field_info
            if isinstance(self, DataSchema):
                annotations[fname] = pyhint(field.hint)
            else:
                annotations[fname] = field.hint

            if parse:
                if (field_parser := field.input_parser) is not None:
                    validator = field.make_pydantic_validator(
                        fname,
                        field_parser,
                        pre=True,
                        allow_none=optional,
                        namespace=namespace,
                    )
                    validators[validator.__name__] = validator
                if (tz := field.extensions.get("tz", None)) is not None:
                    validator = field._pydantic_tz_parser(fname, tz)
                    validators[validator.__name__] = validator
                if (
                    freq := field.extensions.get("freq", None)
                ) is not None and field.type_ == pd.Period:
                    validator = field._pydantic_freq_parser(fname, freq)
                    validators[validator.__name__] = validator
                if dataspec is not None:
                    if (field_parser := dataspec.input_parser) is not None:
                        validator = dataspec.make_pydantic_validator(
                            fname,
                            field_parser,
                            pre=True,
                            allow_none=optional,
                            namespace=field.type_,
                            method_name=fname + "_" + field_parser.__name__,
                        )
                        validators[validator.__name__] = validator
            if validate:
                if dataspec is not None:
                    for field_validator in dataspec.validators:
                        validator = dataspec.make_pydantic_validator(
                            fname,
                            field_validator,
                            allow_none=optional,
                            namespace=field.type_,
                            method_name=fname + "_" + field_validator.__name__,
                        )
                        validators[validator.__name__] = validator
                    if field.validators:
                        # Field validators on Data-typed fields may either operate
                        # on raw data or on the Data instance. If the latter,
                        # conversions back and forth must be applied.
                        if any(
                            getattr(v, "_as_object", False) for v in field.validators
                        ):
                            datatype: type = field.type_

                            def to_object(cls, value):
                                return datatype(value, parse=False, validate=False)

                            def to_data(cls, value):
                                return value.data

                            data_to_object = field.make_pydantic_validator(
                                fname,
                                to_object,
                                converter=True,
                                allow_none=optional,
                                namespace=namespace,
                                method_name="convert_" + fname + "_to_object",
                            )

                            object_to_data = field.make_pydantic_validator(
                                fname,
                                to_data,
                                converter=True,
                                allow_none=optional,
                                namespace=namespace,
                                method_name="convert_" + fname + "_to_data",
                            )

                            data_validators = [
                                field.make_pydantic_validator(
                                    fname, v, allow_none=optional, namespace=namespace
                                )
                                for v in field.validators
                                if not getattr(v, "_as_object", False)
                            ]
                            object_validators = [
                                field.make_pydantic_validator(
                                    fname, v, allow_none=optional, namespace=namespace
                                )
                                for v in field.validators
                                if getattr(v, "_as_object", False)
                            ]
                            field_validators = (
                                data_validators
                                + [data_to_object]
                                + object_validators
                                + [object_to_data]
                            )
                            for v in field_validators:
                                validators[v.__name__] = v
                        else:
                            for field_validator in field.validators:
                                validator = field.make_pydantic_validator(
                                    fname,
                                    field_validator,
                                    allow_none=optional,
                                    namespace=namespace,
                                )
                                validators[validator.__name__] = validator
                else:
                    for field_validator in field.validators:
                        validator = field.make_pydantic_validator(
                            fname,
                            field_validator,
                            allow_none=optional,
                            namespace=namespace,
                        )
                        validators[validator.__name__] = validator
            if optional:
                validator = field._pydantic_nan_parser(fname)
                validators[validator.__name__] = validator
        annotations = self._normalize_annotations(annotations)
        model_class = pydantic_model_class(
            self.name.replace("Schema", "Model"),
            fields=field_infos,
            annotations=annotations,
            validators=validators,
        )

        return model_class

    def model_class(
        self,
        *,
        all_optional: bool = False,
        freeze_settings: bool = True,
        **settings,
    ) -> type[pydantic.BaseModel]:
        """Makes a pydantic validataion model class from self.

        :param all_optional: if True, a None default value is inserted into
            every field that doesn't already defines a default.
        :param freeze_settings: if True, settings are interpreted to
            set fields to constant values; if False, settings are only
            recorded as field default values.
        :param **settings: these keyword arguments will be turned into
        constant fields in the generated model.
        """
        return self._to_pydantic(
            all_optional=all_optional,
            freeze_settings=freeze_settings,
            **settings,
        )

    def _print_dict(self) -> dict[str, Any]:
        """Primitive for to_yaml"""
        attrs = self.__pydantic_model__(**self.__dict__)
        print_attrs = attrs.dict(
            exclude_defaults=True,
            exclude={
                "_hash",
                "input_parser",
                "validators",
                "config",
            },
        )
        print_attrs.setdefault("name", self.name)
        if parser := self.input_parser:
            print_attrs["parser"] = "!" + parser.__name__
        if validators := self.validators:
            keys = auto_label(["custom"] * len(validators))
            validators = {k: "!" + v.__name__ for k, v in zip(keys, validators)}
            print_attrs["validators"] = validators
        if config := self.config:
            print_attrs["config"] = config
        print_attrs["fields"] = {
            name: field.__schema_str__() for name, field in self.fields.items()
        }
        field_sequence = [
            "name",
            "fields",
            "parser",
            "validators",
            "config",
        ]
        return {k: print_attrs[k] for k in field_sequence if k in print_attrs}

    @classmethod
    def to_yaml(cls, representer, node: "Schema"):
        print_dict = node._print_dict()
        return representer.represent_mapping(cls.yaml_tag, print_dict)

    def __str__(self):
        return yaml.dumps(self)


@dataclass
class SchemaParser(Descriptor):
    """A descriptor that registers schema parser methods."""

    mutate_after_registry = True
    name: str
    method: BasicModelParser = dc.field(repr=False)


@dataclass
class SchemaValidator(Descriptor):
    """A descriptor that registers schema validator methods."""

    mutate_after_registry = True
    name: str
    method: BasicModelValidator = dc.field(repr=False)


class IndexSchema(Mapping, DataModelSpecifier):
    """A mapping of index roles to field names."""

    name: ClassVar[str] = "index"
    __index_tags__: ClassVar = {
        "nx_id",
        "nx_key",
        "nx_timestamp",
        "nx_start_time",
        "nx_end_time",
        "nx_period",
        "nx_index",
    }
    _data: frozendict[str, tuple[str]]

    def __init__(self, data: Mapping[str, Iterable], tz=None):
        self._data = frozendict({k: tuple(v) for k, v in data.items()})
        self._tz = tz
        self.validate()

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def validate(self):
        if not all(tag in self.__index_tags__ for tag in self._data):
            msg = f"Index tags can only include {self.__index_tags__}"
            raise ValueError(msg)
        # For the time being, temporal index fields are limited to a single
        # column
        time_tags = ("nx_timestamp", "nx_start_time", "nx_end_time", "nx_period")
        if not all(len(self._data.get(tt, [])) in [0, 1] for tt in time_tags):
            msg = "Temporal indexes are limited to a single column"
            raise ValueError(msg)
        if "nx_id" in self._data:  # Entities only admit secondary indexes
            if not set(self._data) <= {"nx_id", "nx_index"}:
                msg = f"{self} improperly mixes an entity id tag with record tags"
                raise ValueError(msg)
        session_tags = ("nx_start_time", "nx_end_time")
        if any(tag in self._data for tag in session_tags) and not all(
            tag in self._data for tag in session_tags
        ):
            msg = f"{self} must contain either both nx_start_time and nx_end_time keys or none"
            raise ValueError(msg)
        time_tag_alternatives = ("nx_timestamp", "nx_start_time", "nx_period")
        if sum(tag in self._data for tag in time_tag_alternatives) > 1:
            msg = f"{self} features competing temporal tags"
            raise ValueError(msg)

    @classmethod
    def from_fields(cls, *fields: DataField):
        data = defaultdict(list)
        index_tags = {t: True for t in cls.__index_tags__}
        time_tags = ("nx_timestamp", "nx_start_time", "nx_end_time", "nx_period")
        time_zones = set()
        for f in fields:
            tags = {t[0] for t in set(index_tags.items()) & set(f.extensions.items())}
            if len(tags) > 1:
                msg = f"Multiple index tags {tags} in field {f}"
                raise TypeError(msg)
            if tags:
                tag = tags.pop()
                data[tag].append(f.name)
                if tag in time_tags:
                    if field_tz := f.extensions.get("tz"):
                        time_zones.add(field_tz)
        if time_zones:
            if len(time_zones) > 1:
                msg = "Incompatible time zones supplied to index"
                raise ValueError(msg)
            else:
                tz = time_zones.pop()
        else:
            tz = None
        return cls(data, tz=tz)

    def __le__(self, other: "IndexSchema"):
        """Asserts that other extends self."""
        self_items = set(chain(*[[(k, v) for v in l] for k, l in self.items()]))
        other_items = set(chain(*[[(k, v) for v in l] for k, l in other.items()]))
        return self_items <= other_items

    def __ge__(self, other: "IndexSchema"):
        """Asserts that self extends other."""
        self_items = set(chain(*[[(k, v) for v in l] for k, l in self.items()]))
        other_items = set(chain(*[[(k, v) for v in l] for k, l in other.items()]))
        return self_items >= other_items

    def __lt__(self, other: "IndexSchema"):
        return self <= other and self != other

    def __gt__(self, other: "IndexSchema"):
        return self >= other and self != other

    @property
    def is_entity_index(self) -> bool:
        return "nx_id" in self._data

    @property
    def is_record_index(self) -> bool:
        return (
            len(
                set(self._data)
                & {
                    "nx_key",
                    "nx_timestamp",
                    "nx_start_time",
                    "nx_end_time",
                    "nx_period",
                }
            )
            >= 1
        )

    @property
    def is_empty_index(self) -> bool:
        return not self._data

    @property
    def fields(self) -> tuple[str]:
        """Returns a tuple of field names that make up the index."""
        if self.is_empty_index:
            return tuple()
        elif self.is_entity_index:
            return self._data["nx_id"]
        elif self._data.get("nx_timestamp"):
            return self._data["nx_key"] + self._data["nx_timestamp"]
        elif self._data.get("nx_period"):
            return self._data["nx_key"] + self._data["nx_period"]
        elif self._data.get("nx_start_time"):
            return (
                self._data["nx_key"]
                + self._data["nx_start_time"]
                + self._data["nx_end_time"]
            )
        else:
            msg = f"Improperly formed index {self}"
            raise ValueError(msg)

    @property
    def temporal_fields(self) -> tuple[str]:
        """Returns a tuple of field names that define temporal indexing."""
        return tuple(
            chain(
                *[
                    self._data.get(k, ())
                    for k in [
                        "nx_timestamp",
                        "nx_period",
                        "nx_start_time",
                        "nx_end_time",
                    ]
                ]
            )
        )

    @property
    def tz(self):
        return self._tz

    @property
    def nxids(self) -> tuple[str]:
        return self._data.get("nx_identifier", ())

    @property
    def nxkeys(self) -> tuple[str]:
        return self._data.get("nx_key", ())

    @property
    def nxkey(self) -> tuple[str]:
        return self._data.get("nx_key", [None])[0]

    @property
    def nxtime(self) -> Optional[str]:
        return self._data.get("nx_timestamp", [None])[0]

    @property
    def nxstart(self) -> Optional[str]:
        return self._data.get("nx_start_time", [None])[0]

    @property
    def nxend(self) -> Optional[str]:
        return self._data.get("nx_end_time", [None])[0]

    @property
    def nxperiod(self) -> Optional[str]:
        return self._data.get("nx_period", [None])[0]

    def __eq__(self, other: "IndexSchema"):
        try:
            return self._data == other._data
        except AttributeError:
            return False

    def __ne__(self, other: "IndexSchema"):
        try:
            return self._data != other._data
        except AttributeError:
            return False

    def __hash__(self):
        return hash(self._data)


@yaml_object(yaml)
@dataclass
class DataSchema(Schema[DataField]):
    yaml_tag: ClassVar[str] = "!dataschema"
    name: str = dc.field(default="DataSchema")
    fields: dict[str, DataField] = dc.field(default_factory=dict)
    input_parser: Optional[DataModelParser] = dc.field(default=None, repr=False)
    validators: list[DataModelValidator] = dc.field(default_factory=list, repr=False)
    index: IndexSchema = dc.field(init=False, repr=False)
    config: dict = dc.field(default_factory=dict, repr=False)
    _hash: Optional[int] = dc.field(default=None, repr=False)

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        fields: Optional[Iterable[T]] = None,
        parser: Optional[DataModelParser] = None,
        validators: Optional[Iterable[DataModelFilter]] = None,
        **config,
    ):
        fields = list(fields or [])
        super().__init__(
            name=name, fields=fields, parser=parser, validators=validators, **config
        )

    def __post_init__(self):
        self.index = IndexSchema.from_fields(*self.fields.values())
        temporal_fields = [
            self.fields[field_name] for field_name in self.index.temporal_fields
        ]
        if not all(f.nptype.kind == "M" for f in temporal_fields):
            msg = "Temporal index fields must have temporal types"
            raise TypeError(msg)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        try:
            return self._hash == getattr(other, "_hash")
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self._hash != getattr(other, "_hash")
        except AttributeError:
            return False

    def _normalize_annotations(
        self, annotations: Mapping[str, Hint]
    ) -> Mapping[str, Hint]:
        return nxmodel.from_spec(annotations).__annotations__

    @classmethod
    def from_registry(
        cls,
        registry: DescriptorRegistry,
        *,
        new_type_name: str,
        base: Optional["DataSchema"] = None,
    ) -> Optional["DataSchema"]:
        """Returns a data schema from descriptor declarations.

        If no data descriptor is registered, the return value is None.
        """
        if not registry.fetch("data"):
            return base
        fields: Iterable[DataField] = registry.fetch("datafield").values()
        parsers = list(registry.fetch("dataparser").values())
        validators = list(registry.fetch("datavalidator").values())
        if len(parsers) > 1:
            msg = "Only one data parser can be specified"
            raise TypeError(msg)
        elif len(parsers) == 1:
            parser = parsers.pop().method
        else:
            parser = None
        validators = [v.method for v in validators]
        if base is None:
            return cls(
                name=new_type_name, fields=fields, parser=parser, validators=validators
            )
        else:
            return base.extend(
                name=new_type_name, fields=fields, parser=parser, validators=validators
            )

    @classmethod
    def from_modelspec(cls, modelspec: type[nxmodel]) -> "DataSchema":
        try:
            return getattr(modelspec, "dataschema")
        except AttributeError:
            if isinstance(modelspec, (Mapping, TypedDictMeta)):
                return cls.from_fieldmap(modelspec)
            elif dc.is_dataclass(modelspec):
                return cls.from_dataclass(modelspec)
            elif issubclass(modelspec, pydantic.BaseModel):
                return cls.from_pydantic_model_class(modelspec)
            else:
                msg = f"Cannot infer DataSchema from {modelspec}"
                raise TypeError(msg)

    # TODO
    @classmethod
    def from_pydantic_model_class(
        cls, model_cls: type[pydantic.BaseModel]
    ) -> "DataSchema":
        return cls(name=model_cls.__name__)

    # TODO
    @classmethod
    def from_dataclass(cls, model_cls: type[DataClass]) -> "DataSchema":
        return cls(name=model_cls.__name__)

    # TODO
    @classmethod
    def from_fieldmap(
        cls, fieldmap: Union[Mapping[str, type], TypedDictMeta]
    ) -> "DataSchema":
        return cls()

    @classmethod
    def make_pydantic_validator(
        cls_,
        method: DataModelFilter,
        *,
        pre: bool = False,
        namespace: Optional[type] = None,
        method_name: Optional[str] = None,
        docstring: Optional[str] = None,
    ):
        """Returns a method wrapped by pydantic's validator.

        :param method: the validation method, carrying the signature of
            a classmethod validating a single argument.
        :param pre: if True, the method outputs a Pydantic 'pre' validator,
            i.e. a parser.
        :param namespace: an optional type that is passed to the underlying
            method, allowing namespace lookups. If none, the calling class
            (ie. a Pydantic model class) is passed to the method.
        :param method_name: an optional method name that overrides the
            name of the supplied method. This may be necessary to avoid
            namespace conflicts in the generated Pydantic model.
        :param docstring: an optional docstring attached to the validation
            method.
        """
        if pre is True:
            # Parser use case
            def wrapped(cls, values: dict[str, Any]) -> dict[str, Any]:
                if namespace is not None:
                    return method(namespace, values)
                else:
                    return method(cls, values)

        else:
            # Validation use case
            def wrapped(cls, values: dict[str, Any]) -> dict[str, Any]:
                if namespace is not None:
                    assert method(namespace, values), method.__doc__
                else:
                    assert method(cls, values), method.__doc__
                return values

        wrapped.__module__ = method.__module__
        wrapped.__name__ = method_name or method.__name__
        wrapped.__qualname__ = method.__qualname__
        if docstring is not None:
            wrapped.__doc__ = docstring
        else:
            wrapped.__doc__ = method.__doc__
        wrapped.__annotations__ = method.__annotations__

        validator = pydantic.root_validator(pre=pre, allow_reuse=True)(wrapped)
        return validator

    def model_class(
        self,
        *,
        namespace: Optional[type] = None,
        all_optional=False,
        freeze_settings: bool = True,
        parse: bool = True,
        validate: bool = True,
        **settings,
    ) -> type[pydantic.BaseModel]:
        """Makes a pydantic validataion model class from self.

        :param namespace: if a type is supplied, then validation methods can
            look up attributes of that type
        :param all_optional: if True, a None default value is inserted into
            every field that doesn't already defines a default.
        :param parse: if True, the model includes pydantic pre-validators,
            otherwise these are skipped.
        :param validate: if True, the model includes pydantic post-validators,
            otherwise these are skipped.
        :param **settings: these keyword arguments will be turned into
        constant fields in the generated model.
        """
        return self._to_pydantic(
            namespace=namespace,
            all_optional=all_optional,
            freeze_settings=freeze_settings,
            parse=parse,
            validate=validate,
            **settings,
        )

    # field properties

    @property
    def nptypes(self):
        return {name: f.nptype for name, f in self.fields.items()}

    @property
    def is_tabular(self) -> bool:
        return True

    @property
    def is_nested(self) -> bool:
        return not self.is_tabular

    # Indexing properties
    @property
    def is_entity_schema(self) -> bool:
        return self.index.is_entity_index

    @property
    def is_record_schema(self) -> bool:
        return self.index.is_record_index

    @property
    def is_spec_schema(self) -> bool:
        return self.index.is_empty_index

    @property
    def is_indexed(self) -> bool:
        return bool(self.index)

    @property
    def is_identified(self) -> bool:
        return "nx_id" in self.index._data

    @property
    def is_keyed(self) -> bool:
        return "nx_key" in self.index._data

    @property
    def is_temporal(self) -> bool:
        time_tags = ("nx_timestamp", "nx_start_time", "nx_end_time", "nx_period")
        return any(t in self.index._data for t in time_tags)

    @property
    def is_timestamped(self) -> bool:
        return "nx_timestamp" in self.index._data

    @property
    def is_timespanned(self) -> bool:
        session_tags = ("nx_start_time", "nx_end_time")
        return all(t in self.index._data for t in session_tags)

    @property
    def is_timeblocked(self) -> bool:
        return "nx_period" in self.index._data


class TurnToClassMethod:
    method: Callable

    def _post_registry(self):
        """Returns a value that is set on the namespace from which a descriptor was collected."""
        return classmethod(self.method)


class DataParser(DataDescriptor, SchemaParser, TurnToClassMethod):
    handle = "dataparser"
    handle_alternatives = ["dataparsers"]


class DataValidator(DataDescriptor, SchemaValidator, TurnToClassMethod):
    handle = "datavalidator"
    handle_alternatives = ["datavalidators"]


@yaml_object(yaml)
class MetadataSchema(Schema[MetadataField]):
    yaml_tag: ClassVar[str] = "!metadataschema"
    fields: dict[str, MetadataField] = dc.field(default_factory=dict)

    def __post_init__(self):
        self._typespec_fields = {k: v for k, v in self.fields.items() if v.typespec}
        self._objectspec_fields = {k: v for k, v in self.fields.items() if v.objectspec}

    @property
    def typespec_fields(self):
        return self._typespec_fields

    @property
    def objectspec_fields(self):
        return self._objectspec_fields

    def model_class(
        self,
        all_optional: bool = False,
        typespec: Optional[bool] = None,
        objectspec: Optional[bool] = None,
        freeze_settings: bool = False,
        **settings,
    ) -> type[pydantic.BaseModel]:
        """Makes a pydantic validataion model class from self.

        :param all_optional: if True, a None default value is inserted into
            every field that doesn't already defines a default.
        :param typespec: if True, only typespec fields are included in the
            model definition
        :param objectspec: if False, objectspec fields are ignored in the
            model definition
        :param freeze_settings: if True, settings turn field into
            constants. If False, settings only set default values.
        :param **settings: these keyword arguments will be turned into
        constant fields in the generated model.
        """
        if typespec is True:
            return self._to_pydantic(
                all_optional=all_optional,
                freeze_settings=freeze_settings,
                filter=lambda f: f.typespec,
                **settings,
            )
        elif objectspec is False:
            return self._to_pydantic(
                all_optional=all_optional,
                freeze_settings=freeze_settings,
                filter=lambda f: not f.objectspec,
                **settings,
            )
        else:
            return self._to_pydantic(
                all_optional=all_optional, freeze_settings=freeze_settings, **settings
            )


class TurnToStaticMethod:
    method: Callable

    def _post_registry(self):
        """Returns a value that is set on the namespace from which a descriptor was collected."""
        return staticmethod(self.method)


class MetadataParser(SchemaParser, TurnToStaticMethod):
    handle = "metadataparser"
    handle_alternatives = ["metadataparsers"]


class MetadataValidator(SchemaValidator, TurnToStaticMethod):
    handle = "metadatavalidator"
    handle_alternatives = ["metadatavalidators"]


@yaml_object(yaml)
class OptionSchema(Schema[OptionField]):
    yaml_tag: ClassVar[str] = "!optionschema"
    fields: dict[str, OptionField] = dc.field(default_factory=dict)


class OptionParser(SchemaParser, TurnToStaticMethod):
    handle = "optionparser"
    handle_alternatives = ["optionparsers"]


class OptionValidator(SchemaValidator, TurnToStaticMethod):
    handle = "optionvalidator"
    handle_alternatives = ["optionvalidators"]
