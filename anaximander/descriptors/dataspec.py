"""Define the DataSpec class, which is equivalent to a Schema for scalar Data types.

"""
from collections import UserDict
from contextlib import contextmanager
from dataclasses import dataclass
import dataclasses as dc
import datetime as dt
from operator import ge, le, eq
from types import new_class
from typing import Any, Callable, ClassVar, Iterable, Optional, Union, TypedDict

import pandas as pd
import pydantic
from pydantic.fields import FieldInfo

from ..utilities.functions import auto_label, freeze, snake_to_camel
from ..utilities.nxdataclasses import Validated, validate_data, pydantic_model_class
from ..utilities.nxyaml import yaml_object, yaml
from .base import DescriptorRegistry, DataModelSpecifier
from .fieldtypes import UNDEFINED


NoArgCallable = Callable[[], Any]
DataParser = Callable[[type, Any], Any]
DataValidator = Callable[[type, Any], bool]
DataFilter = Union[DataParser, DataValidator]


class PydanticValidatorTypedDict(TypedDict):
    """Type declaration for pydantic validator maps."""

    gt: float
    ge: float
    lt: float
    le: float
    multiple_of: float
    min_items: int
    max_items: int
    min_length: int
    max_length: int
    regex: str


PYDANTIC_VALIDATORS = tuple(PydanticValidatorTypedDict.__annotations__)


class PydanticValidatorMap(UserDict):

    # Operators used in asserting that a validation map extends another
    # This implementation has some deficiencies:
    # * gt and ge specs are not compared, even though technically gt=0 with no
    # ge key ge=0 with no gt key, for instance. Instead, gt and ge are compared
    # independently.
    # * regex comparison is limited to equality

    _xcomparators = {
        "gt": ge,
        "ge": ge,
        "lt": le,
        "le": le,
        "multiple_of": lambda a, b: a / b == a // b,
        "min_items": ge,
        "max_items": le,
        "min_length": ge,
        "max_length": le,
        "regex": eq,  # This is restrictive but an exact solution seems complicated
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        typemap = PydanticValidatorTypedDict.__annotations__
        try:
            for k, v in self.items():
                cls = typemap[k]
                if k in ["min_items", "max_items", "min_length", "max_length"]:
                    assert isinstance(v, cls)
                else:
                    cls(v)
        except (KeyError, TypeError, ValueError, AssertionError):
            raise TypeError

    def __le__(self, other: "PydanticValidatorMap"):
        """Asserts that other extends self."""
        try:
            for key, operator in self._xcomparators.items():
                self_val, other_val = self.get(key), other.get(key)
                if self_val is not None:
                    assert operator(other_val, self_val)
        except (TypeError, AssertionError):
            return False
        else:
            return True

    def __ge__(self, other: "PydanticValidatorMap"):
        """Asserts that self extends other."""
        try:
            for key, operator in self._xcomparators.items():
                self_val, other_val = self.get(key), other.get(key)
                if other_val is not None:
                    assert operator(self_val, other_val)
        except (TypeError, AssertionError):
            return False
        else:
            return True

    def __lt__(self, other: "PydanticValidatorMap"):
        return self <= other and self != other

    def __gt__(self, other: "PydanticValidatorMap"):
        return self >= other and self != other


@yaml_object(yaml)
@dataclass
class DataSpec(Validated, DataModelSpecifier):
    yaml_tag: ClassVar[str] = "!dataspec"

    pytype: type = dc.field()
    name: str = dc.field(default="DataSpec")
    nullable: bool = dc.field(init=False, default=False)
    default: Any = dc.field(default=UNDEFINED, repr=False)
    default_factory: Optional[NoArgCallable] = dc.field(default=None, repr=False)
    const: bool = dc.field(default=False, repr=False)
    input_parser: Optional[DataParser] = dc.field(default=None, repr=False)
    validators: Optional[list[DataValidator]] = dc.field(default=None, repr=False)
    pydantic_validators: dict = dc.field(default_factory=dict, repr=False)
    extensions: dict = dc.field(default_factory=dict, repr=False)
    _hash: Optional[int] = dc.field(default=None, repr=False)

    def __init__(
        self,
        pytype: type,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        default=UNDEFINED,
        default_factory: Optional[NoArgCallable] = None,
        const: bool = False,
        parser: Optional[DataParser] = None,
        validators: Optional[Iterable[DataValidator]] = None,
        gt: float = None,
        ge: float = None,
        lt: float = None,
        le: float = None,
        multiple_of: float = None,
        min_items: int = None,
        max_items: int = None,
        min_length: int = None,
        max_length: int = None,
        regex: str = None,
        **extensions,
    ):
        self.pytype = pytype
        self.name = name or type(self).__name__
        self.nullable = nullable
        self.default = default
        self.default_factory = default_factory
        self.const = const
        self.input_parser = parser
        self.validators = list(validators) if validators is not None else []
        inputs = locals()
        pydantic_validators = {
            k: v for k in PYDANTIC_VALIDATORS if (v := inputs.pop(k, None)) is not None
        }
        self.pydantic_validators = PydanticValidatorMap(pydantic_validators)
        self.extensions = extensions
        self.__post_init__()
        validate_data(self)
        self._hash = self._hash_function()

    def __post_init__(self):
        Validated.__post_init__(self)
        if self.default is not UNDEFINED and self.default_factory is not None:
            msg = "Dataspec cannot simultaneously specify default and default factory"
            raise TypeError(msg)
        elif self.default is UNDEFINED and self.const is True:
            msg = "Field cannot be specified constant without a default value"
            raise TypeError(msg)

    def _hash_function(self):
        attrs = dc.asdict(self)
        del attrs["name"]
        return hash(freeze(attrs))

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
            msg = "Cannot set attributes of DataSpec instances"
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
        pytype: Optional[type] = None,
        *,
        name: Optional[str] = None,
        nullable: Optional[bool] = None,
        default=UNDEFINED,
        default_factory: Optional[NoArgCallable] = None,
        const: Optional[bool] = None,
        parser: Optional[DataParser] = None,
        validators: Optional[Iterable[DataValidator]] = None,
        gt: float = None,
        ge: float = None,
        lt: float = None,
        le: float = None,
        multiple_of: float = None,
        min_items: int = None,
        max_items: int = None,
        min_length: int = None,
        max_length: int = None,
        regex: str = None,
        **extensions,
    ):
        """Returns a new instance with optional attribute modifications"""
        if default is not UNDEFINED and default_factory is not None:
            msg = "Dataspec cannot simultaneously specify default and default factory"
            raise TypeError(msg)
        if pytype is None:
            pytype = self.pytype
        else:
            if not issubclass(pytype, self.pytype):
                msg = f"{self} can only be extended with a subtype of {self.pytype} as its pytype"
                raise TypeError(msg)
        name = name or self.name
        if nullable is None:
            nullable = self.nullable
        if default is UNDEFINED and default_factory is None:
            default = self.default
            default_factory = self.default_factory
        if const is None:
            const = self.const
        elif const is False:
            if self.const is True:
                msg = f"Cannot extend constant {self} with non-constant specification"
                raise TypeError(msg)
        if parser is None:
            parser = self.input_parser
        elif self.input_parser is not None:
            parser = lambda cls, value: self.input_parser(cls, parser(cls, value))
        if validators:
            validators = self.validators + list(validators)
        else:
            validators = self.validators
        inputs = locals()
        new_pydantic_validators = {
            k: v for k in PYDANTIC_VALIDATORS if (v := inputs.pop(k, None)) is not None
        }
        new_pydantic_validators = PydanticValidatorMap(new_pydantic_validators)
        if not new_pydantic_validators:
            pydantic_validators = self.pydantic_validators
        else:
            pydantic_validators = PydanticValidatorMap(
                self.pydantic_validators | new_pydantic_validators
            )
            if not pydantic_validators >= self.pydantic_validators:
                msg = f"{new_pydantic_validators} cannot extend {self.pydantic_validators}"
                raise ValueError(msg)
        extensions = self.extensions | extensions
        kwargs = extensions | dict(pydantic_validators)
        return type(self)(
            pytype,
            name=name,
            nullable=nullable,
            default=default,
            default_factory=default_factory,
            const=const,
            parser=parser,
            validators=validators,
            **kwargs,
        )

    def __le__(self, other: "DataSpec"):
        try:
            assert issubclass(other.pytype, self.pytype)
            assert other.const or not self.const
            assert other.input_parser is not None or self.input_parser is None
            assert all(v in other.validators for v in self.validators)
            assert self.pydantic_validators <= other.pydantic_validators
            assert all(e in other.extensions for e in self.extensions)
        except AssertionError:
            return False
        else:
            return True

    def __lt__(self, other: "DataSpec"):
        return self <= other and self != other

    def __gt__(self, other: "DataSpec"):
        return self >= other and self != other

    def __ge__(self, other: "DataSpec"):
        try:
            assert issubclass(self.pytype, other.pytype)
            assert self.const or not other.const
            assert self.input_parser is not None or other.input_parser is None
            assert all(v in self.validators for v in other.validators)
            assert self.pydantic_validators >= other.pydantic_validators
            assert all(e in self.extensions for e in other.extensions)
        except AssertionError:
            return False
        else:
            return True

    @classmethod
    def make_pydantic_validator(
        cls_,
        fieldname: str,
        method: DataFilter,
        *,
        pre: bool = False,
        allow_none: bool = False,
        namespace: Optional[type] = None,
        method_name: Optional[str] = None,
        docstring: Optional[str] = None,
    ):
        """Returns a method wrapped by pydantic's validator.


        :param fieldname: the name of the field to which the validator
            is applied (passed on to Pydantic).
        :param method: the validation method, carrying the signature of
            a classmethod validating a single argument.
        :param pre: if True, the method outputs a Pydantic 'pre' validator,
            i.e. a parser.
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

        if pre is True:
            # Parser use case
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

    @classmethod
    def _pydantic_tz_parser(
        cls_, fieldname: str, tz: Union[str, dt.timezone, dt.tzinfo]
    ):
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

        method_name = fieldname + "_tz_parser"
        return cls_.make_pydantic_validator(
            fieldname, localize, pre=True, method_name=method_name
        )

    @classmethod
    def _pydantic_freq_parser(cls_, fieldname: str, freq: str):
        """Makes a pre-validator that supplies a datetime with a frequency."""

        @classmethod
        def set_freq(cls, value):
            if isinstance(value, (tuple, pd.Period)):
                return value
            else:
                return (value, freq)

        method_name = fieldname + "_freq_parser"
        return cls_.make_pydantic_validator(
            fieldname, set_freq, pre=True, method_name=method_name
        )

    def model_class(
        self,
        *,
        namespace: Optional[type] = None,
        parse: bool = True,
        validate: bool = True,
    ) -> type[pydantic.BaseModel]:
        """Method for generating single-field Pydantic model classes from self.

        :param namespace: an optional class whose attributes can be looked up
            in validation methods.
        :param parse: if True, self's input parser is turned into a pydantic
            pre-model validator, otherwise it is ignored.
        :param validate: if True, self's validators are turned into pydantic
            model validators, otherwise they are ignored.
        """
        info = dict()
        validators = dict()
        if (default := self.default) is not UNDEFINED:
            info["default"] = default
        if (default_factory := self.default_factory) is not None:
            info["default_factory"] = default_factory
        if (const := self.const) is True:
            info["const"] = const
        if extensions := self.extensions:
            info.update(extensions)
        field_info = (
            FieldInfo(**info, **self.pydantic_validators)
            if validate
            else FieldInfo(**info)
        )
        fields = {"data": field_info}
        if self.nullable:
            annotations = {"data": Optional[self.pytype]}
        else:
            annotations = {"data": self.pytype}
        if parse:
            if (parser := self.input_parser) is not None:
                validator = self.make_pydantic_validator(
                    "data",
                    parser,
                    pre=True,
                    allow_none=self.nullable,
                    namespace=namespace,
                )
                validators[parser.__name__] = validator
            if (tz := self.extensions.get("tz", None)) is not None:
                validator = self._pydantic_tz_parser("data", tz)
                validators[validator.__name__] = validator
            if (freq := self.extensions.get("freq", None)) is not None:
                validator = self._pydantic_freq_parser("data", freq)
                validators[validator.__name__] = validator
        if validate:
            for validator_ in self.validators:
                validator = self.make_pydantic_validator(
                    "data", validator_, allow_none=self.nullable, namespace=namespace
                )
                validators[validator_.__name__] = validator
        model_name = snake_to_camel(self.name)
        model_class = pydantic_model_class(
            model_name,
            fields=fields,
            annotations=annotations,
            validators=validators,
        )

        return model_class

    @classmethod
    def from_registry(
        cls,
        registry: DescriptorRegistry,
        *,
        new_type_name: str,
        base: Optional["DataSpec"] = None,
        **extensions,
    ) -> Optional["DataSpec"]:
        """Returns a DataSpec from descriptor declarations.

        If no data descriptor is registered, the return value is None.
        """
        if not registry.fetch("data"):
            if base is None:
                raise TypeError
            if not extensions:
                return base
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
        if base is not None:
            return base.extend(
                name=new_type_name, parser=parser, validators=validators, **extensions
            )

    def _print_dict(self) -> dict[str, Any]:
        """Primitive for to_yaml"""
        print_attrs = dict()
        print_attrs["pytype"] = self.pytype.__name__
        print_attrs["name"] = self.name
        print_attrs["nullable"] = self.nullable
        if (default := self.default) is not UNDEFINED:
            print_attrs["default"] = default
        if default_factory := self.default_factory:
            print_attrs["default_factory"] = "!" + default_factory.__name__
        if const := self.const:
            print_attrs["const"] = const
        if parser := self.input_parser:
            print_attrs["parser"] = "!" + parser.__name__
        if validators := self.validators:
            keys = auto_label(["custom"] * len(validators))
            validators = {k: "!" + v.__name__ for k, v in zip(keys, validators)}
            print_attrs["validators"] = validators
        if pydantic_validators := self.pydantic_validators:
            for k, v in pydantic_validators.items():
                print_attrs[k] = v
        if extensions := self.extensions:
            print_attrs["extensions"] = extensions
        return print_attrs

    @classmethod
    def to_yaml(cls, representer, node: "DataSpec"):
        print_dict = node._print_dict()
        return representer.represent_mapping(cls.yaml_tag, print_dict)

    def __str__(self):
        return yaml.dumps(self)
