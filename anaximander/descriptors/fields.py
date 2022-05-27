"""This module defines the base descriptor classes."""

from contextlib import contextmanager
from dataclasses import dataclass
import dataclasses as dc
import datetime as dt
from email.mime import base
from types import new_class
import typing
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    ClassVar,
    Type,
    TypeVar,
    Union,
)
import numpy as np
import pandas as pd
import pydantic
from pydantic.fields import FieldInfo
from pyparsing import Opt

from ..utilities.functions import (
    freeze,
    camel_to_snake,
    snake_to_camel,
    auto_label,
)
from ..utilities.nxdataclasses import Validated, pydantic_model_class
from ..utilities.nxyaml import yaml_object, yaml

from .base import Descriptor, DataDescriptor, MetaDescriptor, DataModelSpecifier
from .fieldtypes import UNDEFINED, FieldType, Hint, field_type_from_hint
from .datatypes import nxdata, pyhint, DataFieldType, datafield_type_from_hint
from .attributes import Attribute
from .dataspec import DataSpec, PydanticValidatorMap, PYDANTIC_VALIDATORS


NoneType = type(None)
NoArgCallable = Callable[[], Any]
BasicFieldParser = Callable[[Any], Any]
DataFieldParser = Callable[[type, Any], Any]
BasicFieldValidator = Callable[[Any], bool]
DataFieldValidator = Callable[[type, Any], bool]
BasicFieldFilter = Union[BasicFieldParser, BasicFieldValidator]
DataFieldFilter = Union[DataFieldParser, DataFieldValidator]


T = TypeVar("T", bound=FieldType)


class FieldBase(Generic[T]):
    pass


@yaml_object(yaml)
@dataclass
class Field(Descriptor, Validated, FieldBase):
    """A base class for descriptors that behave like fields.

    A field is any settable attribute with a defined type and optional
    parsing and validation rules. Fields notably include Data and Metadata, as
    well as several others. By contast, there are descriptors that do not
    behave like fields, see for instance MetaCharacters.

    attributes:
        name: the name of the field
        hint: the declared python type hint, e.g. Optional[int]
        type_: the data type of the field, e.g. str or int
        nullable: true if the field is declared Optional, meaning its value can be None
        input_parser: an optional input conversion function. Note that casting to
            type_ is handled automatically by the framework and is better left out of
            the input parser.
        validators: an optional list of field value validation functions
        default: a optional default value. If unspecified and the field is
            optional, the field is considered 'dispensable', which means that
            data models may exclude it -in other words, a data record may
            implement a partial schema, whose fields are a subset of the
            record's class schema. Say for instance:

            @nx.datamodel
            class TempRecord(nx.Record):
                timestamp: datetime = data()
                temperature: float = data()
                comment: Optional[str] = data()

            record = TempRecord(timestamp=..., temperature=...)
            assert record.comment is None
            record.data.comment   # raises AttributeError

            On the other hand, if the comment declaration is written as:

                comment: Optional[str] = data(None)

            Then it will be present in the data as None even if unset. Finally,
            as in the Pydantic inteface, one may use the ellipsis to make a
            field a mandatory constructor argument, so:

                comment: Optional[str] = data(...)

            Means that a record can exist with a None comment, but None must
            be explicitly supplied when building the record or an input
            validation error will be raised.
        default_factory: an optional callable that takes no argument and returns
            a value in case none is supplied. Defining both default and
            default_factory raises a TypeError
        alias: an alternative name that can be summoned in schema or model interfaces
        description: an optional description
        const: if True, the field accepts only its default value
        pydantic_validators: an optional list of validation arguments supported
            by Pydantic's Field interface (e.g. 'gt', 'lt', 'ge', 'le', etc.)
        extensions: an optional dictionary of additional attributes that can
            be added to facilitate integration with functional extensions
            or other libraries.
    """

    # ----- class variables ----- #
    yaml_tag: ClassVar[str] = "!field"

    # ----- constructor attributes ----- #
    default: Any = dc.field(default=UNDEFINED, repr=False)
    default_factory: Optional[NoArgCallable] = dc.field(default=None, repr=False)
    alias: Optional[str] = dc.field(default=None, repr=False)
    title: Optional[str] = dc.field(default=None, repr=False)
    description: Optional[str] = dc.field(default=None, repr=False)
    const: bool = dc.field(default=False, repr=False)
    pydantic_validators: dict = dc.field(default_factory=dict, repr=False)
    extensions: dict = dc.field(default_factory=dict, repr=False)
    _hash: Optional[int] = dc.field(default=None, repr=False)

    # ----- automatic attributes ----- #
    name: str = dc.field(init=False, default=None)
    hint: Any = dc.field(init=False, default=None, repr=False)
    type_: FieldType = dc.field(init=False, default=None)
    nullable: bool = dc.field(init=False, default=False)
    input_parser: Optional[BasicFieldParser] = dc.field(
        init=False, default=None, repr=False
    )
    validators: Optional[list[BasicFieldValidator]] = dc.field(
        init=False, default=None, repr=False
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if self._hash is not None:
            msg = "Cannot set attributes of Field instances"
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

    def __set_name__(self, owner: type, name: str):
        if self._hash is not None:
            return
        super().__set_name__(owner, name)
        if not self.hint:
            hint = self.hint = getattr(owner, "__annotations__", {}).get(name, Any)
            self._set_fieldtype()
        self._hash = self._hash_function()

    def __post_init__(self):
        Validated.__post_init__(self)
        if self.default is not UNDEFINED and self.default_factory is not None:
            msg = "Field cannot simultaneously specify default and default factory"
            raise TypeError(msg)
        elif self.default is UNDEFINED and self.const is True:
            msg = "Field cannot be specified constant without a default value"
            raise TypeError(msg)
        if self.validators is None:
            self.validators = []

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

    @classmethod
    def from_key_value_pair(cls, key, value):
        """Creates a constant field from a key-value pair."""
        field = cls(value, const=True)
        if isinstance(value, np.generic):
            hint = type(value.item())
        else:
            hint = type(value)
        annotation = {key: hint}
        exec_body = lambda ns: ns.update({"__annotations__": annotation})
        pretend_owner = new_class("Owner", exec_body=exec_body)
        field.__set_name__(pretend_owner, key)
        return field

    def copy(
        self, *, name: Optional[str] = None, dtype: Optional[Hint] = None, **kwargs
    ):
        """Returns a new instance, with optional attribute modifications supplied by **kwargs"""
        attr_names = [f.name for f in dc.fields(self) if f.init]
        attrs = {name: getattr(self, name) for name in attr_names}
        attrs.pop("_hash")
        attrs |= kwargs
        copy_ = type(self)(**attrs)
        name = name or self.name
        hint = dtype or self.hint
        annotation = {name: hint}
        exec_body = lambda ns: ns.update({"__annotations__": annotation})
        pretend_owner = new_class("Owner", exec_body=exec_body)
        copy_.__set_name__(pretend_owner, name)
        return copy_

    @classmethod
    def __get_validators__(cls_):
        # Enables type checking by Pydantic in order to validate field creation
        def check_type(cls, value):
            if not isinstance(value, cls_):
                raise TypeError("A Field instance is required")
            return value

        yield check_type

    def _is_optional(self):
        """Examines hint for an Optional flag"""
        origin = typing.get_origin(self.hint)
        args = typing.get_args(self.hint)
        if origin is Union:  # type: ignore
            types = set(args)
            if NoneType in types:
                return True

    def _set_fieldtype(self):
        """Sets the data type from a type hint."""
        if self._is_optional():
            self.nullable = True
        self.type_ = field_type_from_hint(self.hint)

    @property
    def dispensable(self):
        """True if the field is hinted Optional and its default is undefined.

        Dispensable fields are excluded from data interfaces if they are
        unset, but the correponding object attributes will return None.
        For instance, obj.dispensable_field_x would return None if
        that field had not been set on obj. However obj.datadict.["dispensable_field_x"]
        would raise a key error since an unset dispendable field is excluded
        from the data interface.
        If the Optional field only indicates that None is an acceptable value
        and no default should be set because the attribute is mandatory, then
        use the ellipsis as the default value in the Field declaration. This
        positively signals the absence of a default value
        """
        return self.nullable and self.default is UNDEFINED

    def make_pydantic_field_info(
        self,
        setting: Any = UNDEFINED,
        *,
        validators: bool = True,
        optional: bool = False,
        freeze_setting: bool = True,
        **pydantic_validators,
    ) -> FieldInfo:
        """Returns a Pydantic FieldInfo instance.

        :param setting: an optional setting value that can be supplied to override
            the default, and make pydantic field a constant if freeze_setting is
            set to True.
        :param validators: this only applies to the semantic validators
            defined by the pydantic field interface (e.g. gt, lt, etc.), which
            are included if the flag is True (the default), ignored otherwise.
        :param optional: if True, the field is given a None default if there
            is no extant default.
        :param freeze_setting: if True (the default) and a setting is
            supplied, the field is marked as a constant value in the model.
            If False is supplied, this behavior is ignored and the model
            only uses the setting as a default value. Note that the behavior
            is altered if the supplied setting is a class. In that case,
            subclasses of the setting will still be tolerated, hence the
            field is not switched to a constant -but its validation rules
            are redefined. Additionally, for fields of type DataSpec or
            DataSchema (inheriting from DataModelSpecifier), overwriting
            is also tolerated provided that an inequality condition is met.
        :param pydantic_validators: optional additional validators that are
            merged with the field's own validators as applicable.
        """
        kwargs = dict()
        if (default := self.default) is not UNDEFINED:
            kwargs["default"] = default
        if (default_factory := self.default_factory) is not None:
            kwargs["default_factory"] = default_factory
        if self.alias is not None:
            kwargs["alias"] = self.name
        if (title := self.title) is not None:
            kwargs["title"] = title
        if (const := self.const) is True:
            kwargs["const"] = const
        if (description := self.description) is not None:
            kwargs["description"] = description
        if validators:
            field_validators = PydanticValidatorMap(self.pydantic_validators)
            prime_validators = PydanticValidatorMap(pydantic_validators)
            merged_validators = PydanticValidatorMap(
                prime_validators | field_validators
            )
            if not merged_validators >= prime_validators:
                msg = f"{field_validators} cannot extend {prime_validators}"
                raise ValueError(msg)
            if merged_validators:
                kwargs.update(**merged_validators)
        if extensions := self.extensions:
            kwargs.update(extensions)
        # This addresses the case of a field that has already been set,
        # for instance in a parent class
        if setting is not UNDEFINED:
            kwargs["default"] = setting
            kwargs.pop("default_factory", None)
            # If the freeze setting option is True the field is also
            # marked as a constant.
            if freeze_setting:
                # Exceptions are made for type settings, which can always be
                # overloaded with a subtype, and DataModelSpecifier instances
                if isinstance(setting, type):
                    kwargs["nx_basetype"] = setting
                elif isinstance(setting, DataModelSpecifier):
                    kwargs["nx_basespec"] = setting
                else:
                    kwargs["const"] = True
        if optional is True:
            if default_factory is None:
                kwargs.setdefault("default", None)

        return FieldInfo(**kwargs)

    @property
    def parser(self):
        """A method decorator for adding a parsing method."""
        if self.input_parser is not None:
            msg = f"Cannot specify more than one parser for {self}"
            raise TypeError(msg)

        def decorator(method: BasicFieldParser) -> BasicFieldParser:
            with self.modify():
                self.input_parser = method
            return staticmethod(method)

        return decorator

    @property
    def validator(self):
        """A method decorator for adding validation methods."""

        def decorator(method: BasicFieldValidator) -> BasicFieldValidator:
            with self.modify():
                self.validators.append(method)
            return staticmethod(method)

        return decorator

    @classmethod
    def make_pydantic_validator(
        cls_,
        fieldname: str,
        method: BasicFieldFilter,
        *,
        pre: bool = False,
        converter: bool = False,
        allow_none: bool = False,
        namespace: Optional[type] = None,
        method_name: Optional[str] = None,
        docstring: Optional[str] = None,
    ):
        """Returns a method wrapped by pydantic's validator.

        This method is the default behavior for metadata and option fields,
        for which the parser and validator method only take a single value
        as signature. Data field validators are implemented as class method
        and can therefore call on attributes of the validating class.

        :param fieldname: the name of the field to which the validator
            is applied (passed on to Pydantic).
        :param method: the validation method, carrying the signature of
            a classmethod validating a single argument.
        :param pre: if True, the method outputs a Pydantic 'pre' validator,
            i.e. a parser.
        :param converter: this addresses the use case in which the validator
            is generated to convert a data value to a Data object, or the other
            way around. In that case the validator's signature follows the
            parsing form, but it is still declared as a post validator to
            Pydantic. It is illegal to declare pre and converter as True
            simultaneously, but not enforced.
        :allow_none: if True, None is validated without running the method.
        :param namespace: an optional type that is supplied to the validation
            method for parameter lookup. This is featured here for compatibility
            with the DataField.make_pydantic_validator method.
        :param method_name: an optional method name that overrides the
            name of the supplied method. This may be necessary to avoid
            namespace conflicts in the generated Pydantic model.
        :param docstring: an optional docstring attached to the validation
            method.
        """
        if isinstance(method, (classmethod, staticmethod)):
            method = method.__func__

        if pre is True or converter is True:
            # Parser or conversion use case
            def wrapped(cls, value: Any) -> Any:
                if allow_none and value is None:
                    return
                return method(value)

        else:
            # Validation use case
            def wrapped(cls, value: Any) -> Any:
                if allow_none and value is None:
                    return
                assert method(value), method.__doc__
                return value

        wrapped.__module__ = method.__module__
        wrapped.__name__ = method_name or method.__name__
        wrapped.__qualname__ = method.__qualname__
        if docstring is not None:
            wrapped.__doc__ = docstring
        else:
            wrapped.__doc__ = method.__doc__
        wrapped.__doc__ = method.__doc__
        wrapped.__annotations__ = method.__annotations__

        validator = pydantic.validator(fieldname, pre=pre, allow_reuse=True)(wrapped)
        return validator

    @classmethod
    def _pydantic_subtype_validator(cls, name, basetype):
        """Makes a validator that asserts a subclass relationship."""

        @staticmethod
        def is_subtype(value):
            return issubclass(value, basetype)

        docstring = f"{name} must be a subclass of {basetype}"
        return cls.make_pydantic_validator(name, is_subtype, docstring=docstring)

    @classmethod
    def _pydantic_submodel_validator(cls, name, basespec):
        """Makes a validator that asserts a submodel relationship (DataSpec, DataSchema)."""

        @staticmethod
        def is_submodel(value):
            return value >= basespec

        docstring = f"{name} must extend {basespec}"
        return cls.make_pydantic_validator(name, is_submodel, docstring=docstring)

    @classmethod
    def _pydantic_nan_parser(cls, name):
        """Makes a pre-validator that converts np.nan to None."""

        @staticmethod
        def nan_to_none(value):
            try:
                if np.isnan(value):
                    return None
                else:
                    return value
            except TypeError:
                return value

        method_name = name + "_npnan_to_none"
        return cls.make_pydantic_validator(
            name, nan_to_none, pre=True, method_name=method_name
        )

    @classmethod
    def _pydantic_tz_parser(cls, name: str, tz: Union[str, dt.timezone, dt.tzinfo]):
        """Featured here for interface compatibility."""
        return NotImplemented

    @classmethod
    def _pydantic_freq_parser(cls_, name: str, freq: str):
        """Featured here for interface compatibility."""
        return NotImplemented

    def _attribute_retriever(self, source: str):
        """Primitive for retrieving attribute values."""
        if (default := self.default) is UNDEFINED:
            default = None
        target = self.alias or self.name
        return lambda obj: getattr(getattr(obj, source), target, default)

    def _print_dict(self) -> dict[str, Any]:
        """Primitive for to_yaml"""
        attrs = self.__pydantic_model__(**dc.asdict(self))
        print_attrs = attrs.dict(
            exclude_defaults=True,
            exclude={
                "default_factory",
                "pydantic_validators",
                "_hash",
                "hint",
                "type_",
                "input_parser",
                "validators",
                "extensions",
            },
        )
        print_attrs["datatype"] = self.type_.__name__
        if factory := self.default_factory:
            print_attrs["default"] = "!" + factory.__name__
        if parser := self.input_parser:
            print_attrs["parser"] = "!" + parser.__name__
        if std_validators := self.pydantic_validators:
            print_attrs["validators"] = std_validators
        if cst_validators := self.validators:
            validators = print_attrs.setdefault("validators", {})
            keys = auto_label(["custom"] * len(cst_validators))
            for k, v in zip(keys, cst_validators):
                validators[k] = "!" + v.__name__
        if extensions := self.extensions:
            print_attrs["extensions"] = extensions
        field_sequence = [
            "name",
            "title",
            "description",
            "alias",
            "datatype",
            "nullable",
            "default",
            "const",
            "parser",
            "validators",
        ]
        return {k: print_attrs[k] for k in field_sequence if k in print_attrs}

    @classmethod
    def to_yaml(cls, representer, node: "Field"):
        print_dict = node._print_dict()
        return representer.represent_mapping(cls.yaml_tag, print_dict)

    def __schema_str__(self):
        """Print representation inside schemas."""
        print_dict = self._print_dict()
        del print_dict["name"]
        return print_dict

    def __str__(self):
        return yaml.dumps(self)


@yaml_object(yaml)
@dataclass(eq=False)
class DataField(Field, DataDescriptor, DataModelSpecifier):
    yaml_tag: ClassVar[str] = "!datafield"
    handle = "datafield"
    handle_alternatives = ["datafields"]

    type_: DataFieldType = dc.field(init=False, default=None)
    input_parser: Optional[DataFieldParser] = dc.field(
        init=False, default=None, repr=False
    )
    validators: Optional[list[DataFieldValidator]] = dc.field(
        init=False, default=None, repr=False
    )

    @property
    def base_spec(self) -> Optional[DataSpec]:
        """An optional base data specification, if type_ is a Data type."""
        try:
            base_spec_ = getattr(self.type_, "dataspec")
            assert isinstance(base_spec_, DataSpec)
            return base_spec_
        except (AttributeError, AssertionError):
            return None

    def _set_fieldtype(self):
        """Sets the data type from a type hint.

        Currently Optional and typed collections are the only supported non-type hints.
        """
        if self._is_optional():
            self.nullable = True
        self.type_ = datafield_type_from_hint(self.hint)
        if (base_spec := self.base_spec) is not None:
            if base_spec.nullable:
                self.nullable = True
            if base_spec.const:
                self.const = True
            if isinstance(self.default, self.type_):
                # The default is expressed as a DataObject and must be converted to a
                # regular Python type
                self.default = self.default.data
            elif self.default is UNDEFINED:
                if not base_spec.default is UNDEFINED:
                    self.default = base_spec.default
                elif base_spec.default_factory and not self.default_factory:
                    self.default_factory = base_spec.default_factory

    @property
    def pyhint(self):
        """Returns the python type hint corresponding to self's type_."""
        return pyhint(self.type_)

    @property
    def nptype(self):
        """Returns the numpy type corresponding to self's type_."""
        return nxdata.nptype(self.type_)

    @property
    def dtype(self):
        """Returns the pandas dtype corresponding to self's type_."""
        return nxdata.dtype(self.type_, **self.extensions)

    @property
    def parser(self):
        """A method decorator for adding a parsing method."""
        if self.input_parser is not None:
            msg = f"Cannot specify more than one parser for {self}"
            raise TypeError(msg)

        def decorator(method: DataFieldParser) -> DataFieldParser:
            with self.modify():
                self.input_parser = method
            return classmethod(method)

        return decorator

    @property
    def validator(self):
        """A method decorator for adding validation methods."""

        def _decorator(method: DataFieldValidator) -> DataFieldValidator:
            with self.modify():
                self.validators.append(method)
            return classmethod(method)

        def _decorator_as_object(method: DataFieldValidator) -> DataFieldValidator:
            method._as_object = True
            with self.modify():
                self.validators.append(method)
            return classmethod(method)

        def decorator(
            method: Optional[DataFieldValidator] = None, *, as_object: bool = False
        ) -> DataFieldValidator:
            if method is None:
                return _decorator_as_object
            else:
                if as_object:
                    return _decorator_as_object(method)
                else:
                    return _decorator(method)

        return decorator

    @classmethod
    def _pydantic_subtype_validator(cls_, name, basetype):
        """Makes a validator that asserts a subclass relationship."""

        @classmethod
        def is_subtype(cls, value):
            return issubclass(value, basetype)

        docstring = f"{name} must be a subclass of {basetype}"
        return cls_.make_pydantic_validator(name, is_subtype, docstring=docstring)

    @classmethod
    def _pydantic_submodel_validator(cls_, name, basespec):
        """Makes a validator that asserts a submodel relationship (DataSpec, DataSchema)."""

        @classmethod
        def is_submodel(cls, value):
            return value >= basespec

        docstring = f"{name} must extend {basespec}"
        return cls_.make_pydantic_validator(name, is_submodel, docstring=docstring)

    @classmethod
    def _pydantic_nan_parser(cls_, name):
        """Makes a pre-validator that converts np.nan to None."""

        @classmethod
        def nan_to_none(cls, value):
            try:
                if np.isnan(value):
                    return None
                else:
                    return value
            except TypeError:
                return value

        method_name = name + "_npnan_to_none"
        return cls_.make_pydantic_validator(
            name, nan_to_none, pre=True, method_name=method_name
        )

    @classmethod
    def _pydantic_tz_parser(cls_, name: str, tz: Union[str, dt.timezone, dt.tzinfo]):
        """Makes a pre-validator that localizes a datetime type."""

        @classmethod
        def localize(cls, value):
            timestamp = pd.to_datetime(value)
            if (locale := timestamp.tz) is None:
                localized = timestamp.tz_localize(tz)
            elif locale == tz:
                localized = timestamp
            else:
                localized = timestamp.tz_convert(tz)
            return localized.to_pydatetime()

        method_name = name + "_tz_parser"
        return cls_.make_pydantic_validator(
            name, localize, pre=True, method_name=method_name
        )

    @classmethod
    def _pydantic_freq_parser(cls_, name: str, freq: str):
        """Makes a pre-validator that supplies a datetime with a frequency."""

        @classmethod
        def set_freq(cls, value):
            if isinstance(value, (tuple, pd.Period)):
                return value
            else:
                return (value, freq)

        method_name = name + "_freq_parser"
        return cls_.make_pydantic_validator(
            name, set_freq, pre=True, method_name=method_name
        )

    @classmethod
    def make_pydantic_validator(
        cls_,
        fieldname: str,
        method: DataFieldFilter,
        *,
        pre: bool = False,
        converter: bool = False,
        allow_none: bool = False,
        namespace: Optional[type] = None,
        method_name: Optional[str] = None,
        docstring: Optional[str] = None,
    ):
        """Returns a method wrapped by pydantic's validator.

        Data fields implement a different validation signature than
        Metadata or Option fields, which only accept a value as argument. By
        contrast, Data fields also call the model class that validates the
        data and can hence access its attributes -such as metadata attributes.

        :param fieldname: the name of the field to which the validator
            is applied (passed on to Pydantic).
        :param method: the validation method, carrying the signature of
            a classmethod validating a single argument.
        :param pre: if True, the method outputs a Pydantic 'pre' validator,
            i.e. a parser.
        :param converter: this addresses the use case in which the validator
            is generated to convert a data value to a Data object, or the other
            way around. In that case the validator's signature follows the
            parsing form, but it is still declared as a post validator to
            Pydantic. It is illegal to declare pre and converter as True
            simultaneously, but not enforced.
        :allow_none: if True, None is validated without running the method.
        :param namespace: an optional type that is passed to the underlying
            method, allowing namespace lookups. If none, the calling class
            (ie. a Pydantic model class) is passed to the method.
        :param method_name: an optional method name that overrides the
            name of the supplied method. This may be necessary to avoid
            namespace conflicts in the generated Pydantic model.
        :param docstring: an optional docstring attached to the validation
            method.
        """
        if isinstance(method, (classmethod, staticmethod)):
            method = method.__func__

        if pre is True or converter is True:
            # Parser or converter use case
            def wrapped(cls, value: Any) -> Any:
                if allow_none and value is None:
                    return
                if namespace is not None:
                    return method(namespace, value)
                else:
                    return method(cls, value)

        else:
            # Validation use case
            def wrapped(cls, value: Any) -> Any:
                if allow_none and value is None:
                    return
                if namespace is not None:
                    assert method(namespace, value), method.__doc__
                else:
                    assert method(cls, value), method.__doc__
                return value

        wrapped.__module__ = method.__module__
        wrapped.__name__ = method_name or method.__name__
        wrapped.__qualname__ = method.__qualname__
        if docstring is not None:
            wrapped.__doc__ = docstring
        else:
            wrapped.__doc__ = method.__doc__
        wrapped.__doc__ = method.__doc__
        wrapped.__annotations__ = method.__annotations__

        validator = pydantic.validator(fieldname, pre=pre, allow_reuse=True)(wrapped)
        return validator

    def make_pydantic_field_info(
        self,
        setting: Any = UNDEFINED,
        *,
        validators: bool = True,
        optional: bool = False,
        freeze_setting: bool = True,
        **pydantic_validators,
    ) -> FieldInfo:
        if self.base_spec:
            pydantic_validators = (
                self.base_spec.pydantic_validators | pydantic_validators
            )
        return super().make_pydantic_field_info(
            setting,
            validators=validators,
            optional=optional,
            freeze_setting=freeze_setting,
            **pydantic_validators,
        )

    def dataspec(self) -> DataSpec:
        """Makes a DataSpec object from self, in order to spec Data or Series types."""
        if self.base_spec is not None:
            return self.base_spec.extend(
                name=self.name,
                nullable=self._is_optional(),
                default=self.default,
                default_factory=self.default_factory,
                const=self.const if self.const else None,
                parser=self.input_parser,
                validators=self.validators,
                **self.pydantic_validators,
                **self.extensions,
            )
        else:
            return DataSpec(
                nxdata.pytype(self.type_),
                name=self.name,
                nullable=self.nullable,
                default=self.default,
                default_factory=self.default_factory,
                const=self.const,
                parser=self.input_parser,
                validators=self.validators,
                **self.pydantic_validators,
                **self.extensions,
            )

    def set_attribute(
        self,
        host: type,
        *,
        retriever: Optional[Callable] = None,
        transformer: Optional[Callable] = None,
        aggregator: Optional[Callable] = None,
    ):
        retriever = retriever or self._attribute_retriever("_data")
        attribute = Attribute(
            self.name, retriever, transformer, aggregator, _field=self
        )
        setattr(host, self.name, attribute)


@yaml_object(yaml)
@dataclass(eq=False)
class MetadataField(Field, MetaDescriptor):
    handle = "metadata"
    yaml_tag: ClassVar[str] = "!metadatafield"
    typespec: bool = dc.field(default=False)
    objectspec: bool = dc.field(default=False)

    def set_attribute(self, host: type):
        retriever = self._attribute_retriever("_metadata")
        attribute = Attribute(self.name, retriever)
        setattr(host, self.name, attribute)


@yaml_object(yaml)
@dataclass(eq=False)
class OptionField(Field, MetaDescriptor):
    handle = "option"
    handle_alternatives = ["options"]
    yaml_tag: ClassVar[str] = "!optionfield"

    def set_attribute(self, host: type):
        retriever = self._attribute_retriever("_options")
        attribute = Attribute(self.name, retriever)
        setattr(host, self.name, attribute)
