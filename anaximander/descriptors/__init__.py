__all__ = [
    "UNDEFINED",
    "nxdata",
    "nxmodel",
    "metadata",
    "option",
    "data",
    "index",
    "id",
    "key",
    "timestamp",
    "start_time",
    "end_time",
    "period",
    "metaproperty",
    "DataSchema",
    "DataSpec",
    "IndexSchema",
    "DataScope",
    "relation",
    "submodel",
]

from functools import wraps
from typing import Any, Callable, Optional, Union

from .fieldtypes import UNDEFINED
from .datatypes import Hint, nxdata, nxmodel
from .fields import (
    DataField as _DataField,
    MetadataField as _Metadata,
    OptionField as _Option,
)
from .methods import Metaproperty as _Metaproperty, Metamethod as _Metamethod
from .dataspec import DataSpec
from .schema import (
    DataParser as _DataParser,
    DataValidator as _DataValidator,
    MetadataParser as _MetadataParser,
    MetadataValidator as _MetadataValidator,
    OptionParser as _OptionParser,
    OptionValidator as _OptionValidator,
    DataSchema,
    IndexSchema,
)
from .scope import DataScope

NoneType = type(None)
NoArgCallable = Callable[[], Any]
OneArgCallable = Callable[[Any], Any]
Validator = Callable[[Any], None]

_dataspec_keys = [
    "nullable",
    "default",
    "default_factory",
    "const",
]

_pydantic_validator_keys = [
    "gt",
    "ge",
    "lt",
    "le",
    "multiple_of",
    "min_items",
    "max_items",
    "min_length",
    "max_length",
    "allow_mutation",
    "regex",
]


def _transfer_pydantic_validators(inputs: dict):
    pydantic_validators = {
        k: v for k in _pydantic_validator_keys if (v := inputs.pop(k, None)) is not None
    }
    inputs["pydantic_validators"] = pydantic_validators


def data(
    default: Any = UNDEFINED,
    *,
    default_factory: NoArgCallable = None,
    dtype: Hint = None,
    alias: str = None,
    title: str = None,
    description: str = None,
    const: bool = False,
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
    **extensions: Any,
) -> _DataField:
    """The data descriptor declaration function.

    :param default: an optional default value
    :param dtype: an alternative to type hint, useful for the Data archetype
    :param default_factory: alternative to default using a no-argument function
    :param alias: the public name of the field
    :param title: can be any string, used in the schema
    :param description: can be any string, used in the schema
    :param const: if True, only the default value is admissible
    :param gt: only applies to numbers, requires the field to be "greater than"
    :param ge: only applies to numbers, requires the field to be "greater than or equal to"
    :param lt: only applies to numbers, requires the field to be "less than"
    :param le: only applies to numbers, requires the field to be "less than or equal to"
    :param multiple_of: only applies to numbers, requires the field to be "a multiple of"
    :param min_length: only applies to strings, requires the field to have a minimum length
    :param max_length: only applies to strings, requires the field to have a maximum length
    :param regex: only applies to strings, requires the field match agains a regex pattern
    :param **extensions: any additional keyword arguments will be added as is to the schema
    """
    inputs = locals()
    dtype = inputs.pop("dtype")
    _transfer_pydantic_validators(inputs)
    field = _DataField(**inputs)
    if dtype is not None:
        with field.modify():
            field.hint = dtype
            field._set_fieldtype(dtype)
    return field


def _data_parser(method):
    """Method decorator for data parsers."""
    return _DataParser(method.__name__, method)


def _data_validator(method):
    """Method decorator for data validators."""
    return _DataValidator(method.__name__, method)


data.parser = _data_parser
data.validator = _data_validator


def metadata(
    default: Any = UNDEFINED,
    *,
    typespec: bool = False,
    objectspec: bool = False,
    default_factory: NoArgCallable = None,
    alias: str = None,
    title: str = None,
    description: str = None,
    const: bool = False,
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
    **extensions: Any,
) -> _Metadata:
    """The metadata descriptor declaration function.

    :param default: an optional default value
    :param typespec: if True, a part of the type specification, i.e. type-level only
    :param default_factory: alternative to default using a no-argument function
    :param alias: the public name of the field
    :param title: can be any string, used in the schema
    :param description: can be any string, used in the schema
    :param const: if True, only the default value is admissible
    :param gt: only applies to numbers, requires the field to be "greater than"
    :param ge: only applies to numbers, requires the field to be "greater than or equal to"
    :param lt: only applies to numbers, requires the field to be "less than"
    :param le: only applies to numbers, requires the field to be "less than or equal to"
    :param multiple_of: only applies to numbers, requires the field to be "a multiple of"
    :param min_length: only applies to strings, requires the field to have a minimum length
    :param max_length: only applies to strings, requires the field to have a maximum length
    :param regex: only applies to strings, requires the field match agains a regex pattern
    :param **extensions: any additional keyword arguments will be added as is to the schema
    """
    inputs = locals()
    _transfer_pydantic_validators(inputs)
    return _Metadata(**inputs)


def _metadata_parser(method):
    """Method decorator for metadata parsers."""
    return _MetadataParser(method.__name__, method)


def _metadata_validator(method):
    """Method decorator for metadata validators."""
    return _MetadataValidator(method.__name__, method)


metadata.parser = _metadata_parser
metadata.validator = _metadata_validator


def option(
    default: Any = UNDEFINED,
    *,
    default_factory: NoArgCallable = None,
    alias: str = None,
    title: str = None,
    description: str = None,
    const: bool = False,
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
    **extensions: Any,
) -> _Option:
    """The option descriptor declaration function.

    :param default: an optional default value
    :param default_factory: alternative to default using a no-argument function
    :param alias: the public name of the field
    :param title: can be any string, used in the schema
    :param description: can be any string, used in the schema
    :param const: if True, only the default value is admissible
    :param gt: only applies to numbers, requires the field to be "greater than"
    :param ge: only applies to numbers, requires the field to be "greater than or equal to"
    :param lt: only applies to numbers, requires the field to be "less than"
    :param le: only applies to numbers, requires the field to be "less than or equal to"
    :param multiple_of: only applies to numbers, requires the field to be "a multiple of"
    :param min_length: only applies to strings, requires the field to have a minimum length
    :param max_length: only applies to strings, requires the field to have a maximum length
    :param regex: only applies to strings, requires the field match agains a regex pattern
    :param **extensions: any additional keyword arguments will be added as is to the schema
    """
    inputs = locals()
    _transfer_pydantic_validators(inputs)
    return _Option(**inputs)


def _option_parser(method):
    """Method decorator for option parsers."""
    return _OptionParser(method.__name__, method)


def _option_validator(method):
    """Method decorator for option validators."""
    return _OptionValidator(method.__name__, method)


option.parser = _option_parser
option.validator = _option_validator

# Indexing field declaration functions


def index_field_declarator(
    tag: str, *, first_line_doc: Optional[str] = None
) -> Callable:
    @wraps(data)
    def field_wrapper(
        default: Any = UNDEFINED,
        *,
        default_factory: NoArgCallable = None,
        dtype: Hint = None,
        alias: str = None,
        title: str = None,
        description: str = None,
        const: bool = False,
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
        **extensions: Any,
    ) -> _DataField:
        extensions = extensions | {tag: True}
        return data(
            default,
            default_factory=default_factory,
            dtype=dtype,
            alias=alias,
            title=title,
            description=description,
            const=const,
            gt=gt,
            ge=ge,
            lt=lt,
            le=le,
            multiple_of=multiple_of,
            min_items=min_items,
            max_items=max_items,
            min_length=min_length,
            max_length=max_length,
            regex=regex,
            **extensions,
        )

    if first_line_doc is not None:
        documentation = data.__doc__[data.__doc__.index("\n\n") + 2 :]
        documentation = first_line_doc + "\n\n" + documentation
        field_wrapper.__doc__ = documentation
    return field_wrapper


index = index_field_declarator(
    "nx_index", first_line_doc="Declarator for an indexed data field"
)

id = index_field_declarator("nx_id", first_line_doc="Declarator for entity identifiers")

key = index_field_declarator("nx_key", first_line_doc="Declarator for recrord keys")

timestamp = index_field_declarator(
    "nx_timestamp", first_line_doc="Declarator for record timestamps"
)

start_time = index_field_declarator(
    "nx_start_time", first_line_doc="Declarator for session record start time"
)

end_time = index_field_declarator(
    "nx_end_time", first_line_doc="Declarator for session record end time"
)

period = index_field_declarator(
    "nx_period", first_line_doc="Declarator for journal record period"
)


def _metaproperty(method: Callable, *metacharacters: str) -> _Metaproperty:
    """The metaproperty decorating function."""
    return _Metaproperty(method, *metacharacters)


def metaproperty(positional: Union[Callable, str], *metacharacters: str):
    """Generates a method decorator that declares a metaproperty."""

    if isinstance(positional, Callable):
        return _Metaproperty(positional)
    elif isinstance(positional, str):
        metacharacters = positional, *metacharacters

        def decorator(method: Callable):
            return _metaproperty(method, *metacharacters)

        return decorator
    else:
        msg = "Incorrect arguments supplied to metaproperty decorator"
        raise TypeError(msg)


def metamethod(method: Callable[..., Callable]):
    """Method decorator that declares a metamethod."""
    return _Metamethod(method=method)


def relation(datatype: type, **kwargs):
    pass


def submodel(**kwargs):
    pass
