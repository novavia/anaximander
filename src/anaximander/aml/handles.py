"""Handle objects for AML declarators and metadata bindings."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


from collections.abc import Callable, Mapping
from functools import update_wrapper
from numbers import Real
from typing import Any, ClassVar, Literal, cast

from ..utils.meta import Singleton
from .declarative import (
    DECLARATIVE_NAMESPACE,
    DeclarativeNamespace,
)
from .declarators import (
    MISSING,
    AssignableMetadescriptor,
    BackLinkProtodescriptor,
    DataProtodescriptor,
    Declarator,
    EnumerationCallableDeclarator,
    FieldProtodescriptor,
    LinkProtodescriptor,
    MetadataDeclarator,
    MetadataValidator,
    Missing,
    NxFieldDeclarator,
    NxFieldValidator,
    OptionDeclarator,
    OptionValidator,
    ParserDeclarator,
    ValidatorDeclarator,
)

# endregion

# =============================================================================
# Decorator helpers
# =============================================================================
# region Decorator helpers


def _decorator_factory(
    *members_or_fn: Any,
    declarator_type: type[EnumerationCallableDeclarator[Callable[..., Any]]],
    err_label: str,
) -> Any:
    """Build a declarator or decorator for enumeration callables."""
    if members_or_fn and callable(members_or_fn[0]):
        if len(members_or_fn) != 1:
            raise TypeError(f"{err_label} decorator cannot mix members and callable.")
        fn = members_or_fn[0]
        doc = MISSING if fn.__doc__ is None else fn.__doc__
        declarator = declarator_type(callable=fn, members=(), doc=doc)
        update_wrapper(fn, declarator, updated=(), assigned=())  # type: ignore[arg-type]
        return declarator
    members = tuple(members_or_fn)

    def decorator(fn: Callable) -> Declarator:
        doc = MISSING if fn.__doc__ is None else fn.__doc__
        declarator = declarator_type(callable=fn, members=members, doc=doc)
        update_wrapper(fn, declarator, updated=(), assigned=())  # type: ignore[arg-type]
        return declarator

    return decorator

# endregion

# =============================================================================
# Handle classes
# =============================================================================
# region Handle classes


class DeclaratorHandle[DT: Declarator](metaclass=Singleton):
    """Base class for AML declarator handles.

    Instances act as lightweight facades for creating declarators and registering
    declarations or bindings in the active declarative class body.
    """

    __handle__: ClassVar[str]
    __declarator_type__: type[DT]

    def __init_subclass__(cls) -> None:
        try:
            if "__handle__" in cls.__dict__:
                cls.__declarator_type__ = Declarator.__handles__[cls.__handle__]
        except KeyError:
            raise ValueError(f"Invalid handle '{cls.__handle__}' for declarator handle.")

    @property
    def declarator_type(self) -> type[DT]:
        """The declarator type for this handle."""
        return self.__class__.__declarator_type__

    @property
    def handle(self) -> str:
        """The handle name."""
        return self.__class__.__handle__

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__"):
            object.__setattr__(self, name, value)
            return
        self.bind(name, value)

    def _require_namespace(self) -> DeclarativeNamespace:
        dns = DECLARATIVE_NAMESPACE.get()
        if dns is None:
            raise RuntimeError(f"{self.handle} handle can only be used in declarative bodies.")
        return dns

    def bind(self, name: str, value: Any) -> None:
        """Register a binding for a previously declared declarator."""
        dns = self._require_namespace()
        dns.register_binding(name, value, handle=self.handle)


class AssignableMetadescriptorHandle[DT](DeclaratorHandle[AssignableMetadescriptor]):
    """Handle for metadescriptor declarators and bindings."""
    __validator_type__: type[MetadataValidator | OptionValidator | NxFieldValidator]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__"):
            object.__setattr__(self, name, value)
        else:
            self.bind(name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        if self.handle not in {"metadata", "option", "nxfield"}:
            raise KeyError(f"{self.handle} handle does not support item assignment.")
        self.bind(name, value)

    def _declare(
            self,
            name: str, *,
            type: type,
            nullable: bool = False,
            default: Any = MISSING,
            factory: Callable[[], Any] | Missing = MISSING,
            validator: Callable[[Any], bool] | Missing = MISSING,
            doc: str | Missing = MISSING,
            config: Mapping[str, Any] | Missing = MISSING
    ) -> AssignableMetadescriptor:
        """Declare a named declarator without binding it as a class attribute."""
        dns = self._require_namespace()
        declarator = self.__declarator_type__(
            default=default,
            factory=factory,
            validator=validator,
            doc=doc,
            config=config,
        )
        classvar = True
        annotation = (
            f"ClassVar[{type.__name__} | None]" if nullable else f"ClassVar[{type.__name__}]"
        )
        declarator.__set_type__(
            annotation=annotation, type_=type, classvar=classvar, nullable=nullable
        )
        dns.register_declarator(declarator, name=name)
        return declarator

    @property
    def validator_type(self) -> type[MetadataValidator | OptionValidator | NxFieldValidator]:
        """The validator declarator type for this handle, if any."""
        return self.__validator_type__

    def validator(self, *members_or_fn):
        """Create a validation declarator as a decorator."""
        if (validator_type := self.validator_type) is None:
            raise TypeError(f"{self.handle} handle does not support validators.")
        return _decorator_factory(
            *members_or_fn, declarator_type=validator_type, err_label="Validator"
        )


class MetadataHandle(AssignableMetadescriptorHandle[MetadataDeclarator]):
    """Handle for metadata declarators and bindings."""
    __handle__ = "metadata"
    __validator_type__ = MetadataValidator

    def __call__(
            self,
            default: Any = MISSING,
            factory: Callable[[], Any] | Missing = MISSING,
            validator: Callable[[Any], bool] | Missing = MISSING,
            doc: str | Missing = MISSING,
            config: Mapping[str, Any] | Missing = MISSING,
    ) -> Any:
        """Declares domain metadata for use in class assignments."""
        return MetadataDeclarator(
            domain=True,
            default=default,
            factory=factory,
            validator=validator,
            doc=doc,
            config=config,
        )

    def declare(
            self,
            name: str, *,
            type: type,
            nullable: bool = False,
            default: Any = MISSING,
            factory: Callable[[], Any] | Missing = MISSING,
            validator: Callable[[Any], bool] | Missing = MISSING,
            doc: str | Missing = MISSING,
            config: Mapping[str, Any] | Missing = MISSING
    ) -> MetadataDeclarator:
        """Declare a named metadata declarator with a fixed type."""
        declarator = cast(
            MetadataDeclarator,
            super()._declare(
                name,
                type=type,
                nullable=nullable,
                default=default,
                factory=factory,
                validator=validator,
                doc=doc,
                config=config,
            ),
        )
        declarator._set_once("domain", False)
        return declarator


class OptionHandle(AssignableMetadescriptorHandle[OptionDeclarator]):
    """Handle for option declarators and bindings."""
    __handle__ = "option"
    __validator_type__ = OptionValidator

    def declare(
            self,
            name: str, *,
            type: type,
            nullable: bool = False,
            default: Any = MISSING,
            factory: Callable[[], Any] | Missing = MISSING,
            validator: Callable[[Any], bool] | Missing = MISSING,
            doc: str | Missing = MISSING,
            config: Mapping[str, Any] | Missing = MISSING
    ) -> OptionDeclarator:
        """Declare a named nxfield with a fixed string type."""
        declarator = cast(
            OptionDeclarator,
            super()._declare(
                name,
                type=type,
                nullable=nullable,
                default=default,
                factory=factory,
                validator=validator,
                doc=doc,
                config=config,
            ),
        )
        return declarator


class NxFieldHandle(AssignableMetadescriptorHandle[NxFieldDeclarator]):
    """Handle for nxfield declarators and bindings."""
    __handle__ = "nxfield"
    __validator_type__ = NxFieldValidator

    def declare(
            self,
            name: str, *,
            fieldtype: type,
            doc: str | Missing = MISSING,
            config: Mapping[str, Any] | Missing = MISSING
    ) -> NxFieldDeclarator:
        """Declare a named nxfield with a fixed string type."""
        declarator = cast(
            NxFieldDeclarator,
            super()._declare(
                name,
                type=str,
                doc=doc,
                config=config,
            ),
        )
        declarator._set_once("fieldtype", fieldtype)
        return declarator


class FieldHandle[DT](DeclaratorHandle[FieldProtodescriptor]):
    """Abstract base class for field handles exposing validators."""
    __validator_type__ = ValidatorDeclarator

    @property
    def validator_type(self) -> type[ValidatorDeclarator]:
        """The validator declarator type for this handle, if any."""
        return self.__validator_type__

    def validator(self, *members_or_fn):
        """Create a validation declarator as a decorator."""
        if (validator_type := self.validator_type) is None:
            raise TypeError(f"{self.handle} handle does not support validators.")
        return _decorator_factory(
            *members_or_fn, declarator_type=validator_type, err_label="Validator"  # type: ignore
        )


class DataHandle(FieldHandle[DataProtodescriptor]):
    """Handle for data protodescriptor constructors."""
    __handle__ = "data"

    def __call__(
        self,
        default: Any = MISSING,
        *,
        factory: Callable[[], Any] | Missing = MISSING,
        typekey: bool = False,
        required: bool = False,
        load: Literal["eager", "lazy"] | Missing = MISSING,
        unique: bool = False,
        index: bool = False,
        key: bool = False,
        sequence: bool = False,
        timestamp: bool = False,
        start_time: bool = False,
        end_time: bool = False,
        period: bool = False,
        location: bool = False,
        geom: bool = False,
        validator: Callable[[Any], bool] | Missing = MISSING,
        gt: Real | Missing = MISSING,
        ge: Real | Missing = MISSING,
        lt: Real | Missing = MISSING,
        le: Real | Missing = MISSING,
        min_length: int | Missing = MISSING,
        max_length: int | Missing = MISSING,
        pattern: str | Missing = MISSING,
        repr: bool | Callable | str | Missing = MISSING,
        doc: str | Missing = MISSING,
        config: Mapping[str, Any] | Missing = MISSING,
    ) -> Any:
        """Construct a data protodescriptor with current AML field semantics."""
        return DataProtodescriptor(
            default=default,
            factory=factory,
            typekey=typekey,
            required=required,
            load=load,
            unique=unique,
            index=index,
            key=key,
            sequence=sequence,
            timestamp=timestamp,
            start_time=start_time,
            end_time=end_time,
            period=period,
            location=location,
            geom=geom,
            validator=validator,
            gt=gt,
            ge=ge,
            lt=lt,
            le=le,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            repr=repr,
            doc=doc,
            config=config,
        )

    def parser(self, *members_or_fn):
        """Create a field parser declarator as a decorator."""
        return _decorator_factory(
            *members_or_fn, declarator_type=ParserDeclarator, err_label="Parser"  # type: ignore
        )


class LinkHandle(FieldHandle[LinkProtodescriptor]):
    """Handle for link protodescriptor constructors."""
    __handle__ = "link"

    def __call__(
        self,
        *,
        required: bool = False,
        on_delete: Literal["restrict", "set_null", "cascade"] = "restrict",
        load: Literal["eager", "lazy"] | Missing = MISSING,
        unique: bool = False,
        key: bool = False,
        validator: Callable[[Any], bool] | Missing = MISSING,
        repr: bool | Callable | str | Missing = MISSING,
        doc: str | Missing = MISSING,
        config: Mapping[str, Any] | Missing = MISSING,
    ) -> Any:
        """Construct a link protodescriptor with current AML field semantics."""
        return LinkProtodescriptor(
            required=required,
            load=load,
            unique=unique,
            key=key,
            on_delete=on_delete,
            validator=validator,
            repr=repr,
            doc=doc,
            config=config,
        )


class BacklinkHandle(DeclaratorHandle[BackLinkProtodescriptor]):
    """Handle for backlink protodescriptor constructors."""
    __handle__ = "backlink"

    def __call__(
        self,
        *,
        load: Literal["eager", "lazy"] | Missing = MISSING,
        unique: bool = False,
        via: type | Missing = MISSING,
        limit: int | Missing = MISSING,
        repr: bool | Callable | str | Missing = MISSING,
        doc: str | Missing = MISSING,
        config: Mapping[str, Any] | Missing = MISSING,
    ) -> Any:
        """Construct a backlink protodescriptor with current AML field semantics."""
        return BackLinkProtodescriptor(
            load=load,
            unique=unique,
            via=via,
            limit=limit,
            repr=repr,
            doc=doc,
            config=config,
        )


class ParserHandle(DeclaratorHandle[ParserDeclarator]):
    """Handle for parser declarators."""
    __handle__ = "parser"

    def __call__(self, *members_or_fn):
        """Create a parser declarator as a decorator."""
        return _decorator_factory(
            *members_or_fn, declarator_type=ParserDeclarator, err_label="Parser"  # type: ignore
        )


class ValidatorHandle(DeclaratorHandle[ValidatorDeclarator]):
    """Handle for validator declarators."""
    __handle__ = "validator"

    def __call__(self, *members_or_fn):
        """Create a validator declarator as a decorator."""
        return _decorator_factory(
            *members_or_fn, declarator_type=ValidatorDeclarator, err_label="Validator"  # type: ignore
        )

# endregion

# =============================================================================
# Singleton instances
# =============================================================================
# region Singleton instances


metadata = MetadataHandle()
option = OptionHandle()
nxfield = NxFieldHandle()
data = DataHandle()
link = LinkHandle()
backlink = BacklinkHandle()
parser = ParserHandle()
validator = ValidatorHandle()
meta = metadata

# endregion
