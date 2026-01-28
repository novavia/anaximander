"""Interface objects for AML declarators and metadata bindings."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


from collections.abc import Mapping
from functools import update_wrapper
from numbers import Real
from typing import Any, Callable, Literal

from ..utils.meta import Singleton
from .declarative import (
    DECLARATIVE_NAMESPACE,
    MISSING,
    DeclarativeNamespace,
    Declarator,
    Missing,
    declarative,
)
from .metadescriptors import (
    AssignableMetadescriptor,
    MetadataDeclarator,
    MetadataValidator,
    NxFieldDeclarator,
    NxFieldValidator,
    OptionDeclarator,
    OptionValidator,
)
from .protodescriptors import (
    BackLinkProtodescriptor,
    DataProtodescriptor,
    FieldProtodescriptor,
    LinkProtodescriptor,
    ParserDeclarator,
    ValidatorDeclarator,
)
from .prototype import Arche, prototype

# endregion

# =============================================================================
# Interface classes
# =============================================================================
# region Interface classes


class DeclaratorInterface[DT: Declarator](metaclass=Singleton):
    """Base class for AML declarator interfaces.

    Instances act as lightweight facades for creating declarators and registering
    declarations or bindings in the active declarative class body.
    """

    __handle__: str
    __declarator_type__: type[DT]

    def __init_subclass__(cls) -> None:
        try:
            if "__handle__" in cls.__dict__:
                cls.__declarator_type__ = Declarator.__handles__[cls.__handle__]
        except KeyError:
            raise ValueError(f"Invalid handle '{cls.__handle__}' for declarator interface.")

    @property
    def declarator_type(self) -> type[DT]:
        """The declarator type for this interface."""
        return self.__class__.__declarator_type__

    @property
    def handle(self) -> str:
        """The interface handle."""
        return self.__class__.__handle__

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__"):
            object.__setattr__(self, name, value)
            return
        self.bind(name, value)

    def _require_namespace(self) -> DeclarativeNamespace:
        dns = DECLARATIVE_NAMESPACE.get()
        if dns is None:
            raise RuntimeError(f"{self.handle} interface can only be used in declarative bodies.")
        return dns

    def bind(self, name: str, value: Any) -> None:
        """Register a binding for a previously declared declarator."""
        dns = self._require_namespace()
        dns.register_binding(name, value, handle=self.handle)


class AssignableMetadescriptorInterface[DT](DeclaratorInterface[AssignableMetadescriptor]):
    """Interface for metadescriptor declarators and bindings."""
    __validator_type__: type[MetadataValidator | OptionValidator | NxFieldValidator]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__"):
            object.__setattr__(self, name, value)
            return
        if name in self._global_names():
            self.bind(name, value)
            return
        super().__setattr__(name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        if self.handle not in {"metadata", "option", "nxfield"}:
            raise KeyError(f"{self.handle} interface does not support item assignment.")
        self.bind(name, value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if name in self._global_names():
            return name
        raise AttributeError(f"{self.handle} interface has no attribute '{name}'.")

    def __dir__(self) -> list[str]:
        entries = set(super().__dir__())
        entries.update(self._global_names())
        return sorted(entries)

    def _global_names(self) -> set[str]:
        from .declarative import global_names
        return global_names(self.handle)

    def _register_known_name(self, name: str) -> None:
        from .declarative import register_global_name
        if self.handle in {"metadata", "option", "nxfield"}:
            register_global_name(self.handle, name)
            if not hasattr(self.__class__, name):
                setattr(self.__class__, name, name)
            if not hasattr(self, name):
                object.__setattr__(self, name, name)

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
        if isinstance(declarator, MetadataDeclarator):
            declarator._set_once("domain", False)
        classvar = True
        annotation = (
            f"ClassVar[{type.__name__} | None]" if nullable else f"ClassVar[{type.__name__}]"
        )
        declarator.__set_type__(
            annotation=annotation, type_=type, classvar=classvar, nullable=nullable
        )
        dns.register_declaration(declarator, name=name)
        return declarator

    @property
    def validator_type(self) -> type[MetadataValidator | OptionValidator | NxFieldValidator]:
        """The validator declarator type for this interface, if any."""
        return self.__validator_type__

    def validator(self, *members: str) -> Callable[[Callable[[declarative, Any], bool]], MetadataValidator | OptionValidator | NxFieldValidator]:  # noqa
        """Create a validation declarator as a decorator."""
        if (validator_type := self.validator_type) is None:
            raise TypeError(f"{self.handle} interface does not support validators.")

        def decorator(fn: Callable[[declarative, Any], bool]) -> MetadataValidator | OptionValidator | NxFieldValidator:  # noqa
            doc = MISSING if fn.__doc__ is None else fn.__doc__
            declarator = validator_type(callable=fn, members=members, doc=doc)
            update_wrapper(declarator, fn, updated=())  # type: ignore[arg-type]
            return declarator

        return decorator


class MetadataInterface(AssignableMetadescriptorInterface[MetadataDeclarator]):
    """Interface for metadata declarators and bindings."""
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


class OptionInterface(AssignableMetadescriptorInterface[OptionDeclarator]):
    """Interface for option declarators and bindings."""
    __handle__ = "option"
    __validator_type__ = OptionValidator


class NxFieldInterface(AssignableMetadescriptorInterface[NxFieldDeclarator]):
    """Interface for nxfield declarators and bindings."""
    __handle__ = "nxfield"
    __validator_type__ = NxFieldValidator


class FieldInterface[DT](DeclaratorInterface[FieldProtodescriptor]):
    """Abstract base class for field interfaces exposing validators."""
    __validator_type__ = ValidatorDeclarator

    @property
    def validator_type(self) -> type[ValidatorDeclarator]:
        """The validator declarator type for this interface, if any."""
        return self.__validator_type__

    def validator(self, *members: str) -> Callable[[Callable[[Arche | prototype, Any], bool]], ValidatorDeclarator]:  # noqa
        """Create a validation declarator as a decorator."""
        if (validator_type := self.validator_type) is None:
            raise TypeError(f"{self.handle} interface does not support validators.")

        def decorator(fn: Callable[[Arche | prototype, Any], bool]) -> ValidatorDeclarator:
            doc = MISSING if fn.__doc__ is None else fn.__doc__
            declarator = validator_type(callable=fn, members=members, doc=doc)
            update_wrapper(declarator, fn, updated=())  # type: ignore[arg-type]
            return declarator

        return decorator


class DataInterface(FieldInterface[DataProtodescriptor]):
    """Interface for data protodescriptor constructors."""
    __handle__ = "data"

    def __call__(
        self,
        default: Any = MISSING,
        *,
        factory: Callable[[], Any] | Missing = MISSING,
        unique: bool = False,
        index: bool = False,
        required: bool = False,
        typekey: bool = False,
        key: bool = False,
        sequence: bool = False,
        timestamp: bool = False,
        start_time: bool = False,
        end_time: bool = False,
        period: bool = False,
        location: bool = False,
        geom: bool = False,
        load: Literal["eager", "lazy"] | Missing = MISSING,
        repr: bool | Callable | str | Missing = MISSING,
        validator: Callable[[Any], bool] | Missing = MISSING,
        gt: Real | Missing = MISSING,
        ge: Real | Missing = MISSING,
        lt: Real | Missing = MISSING,
        le: Real | Missing = MISSING,
        min_length: int | Missing = MISSING,
        max_length: int | Missing = MISSING,
        pattern: str | Missing = MISSING,
        doc: str | Missing = MISSING,
        config: Mapping[str, Any] | Missing = MISSING,
    ) -> Any:
        """Construct a data protodescriptor with current AML field semantics."""
        return DataProtodescriptor(
            default=default,
            factory=factory,
            validator=validator,
            load=load,
            repr=repr,
            unique=unique,
            index=index,
            required=required,
            typekey=typekey,
            key=key,
            sequence=sequence,
            timestamp=timestamp,
            start_time=start_time,
            end_time=end_time,
            period=period,
            location=location,
            geom=geom,
            gt=gt,
            ge=ge,
            lt=lt,
            le=le,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            doc=doc,
            config=config,
        )

    def parser(self, *members: str) -> Callable[[Callable[[Arche | prototype, Any], bool]], ParserDeclarator]:  # noqa
        """Create a field parser declarator as a decorator."""
        def decorator(fn: Callable[[Arche | prototype, Any], bool]) -> ParserDeclarator:
            doc = MISSING if fn.__doc__ is None else fn.__doc__
            declarator = ParserDeclarator(callable=fn, members=members, doc=doc)
            update_wrapper(declarator, fn, updated=())  # type: ignore[arg-type]
            return declarator

        return decorator


class LinkInterface(FieldInterface[LinkProtodescriptor]):
    """Interface for link protodescriptor constructors."""
    __handle__ = "link"

    def __call__(
        self,
        *,
        unique: bool = False,
        required: bool = False,
        key: bool = False,
        on_delete: Literal["restrict", "set_null", "cascade"] = "restrict",
        load: Literal["eager", "lazy"] | Missing = MISSING,
        repr: bool | Callable | str | Missing = MISSING,
        validator: Callable[[Any], bool] | Missing = MISSING,
        doc: str | Missing = MISSING,
        config: Mapping[str, Any] | Missing = MISSING,
    ) -> Any:
        """Construct a link protodescriptor with current AML field semantics."""
        return LinkProtodescriptor(
            unique=unique,
            key=key,
            on_delete=on_delete,
            required=required,
            load=load,
            repr=repr,
            validator=validator,
            doc=doc,
            config=config,
        )


class BacklinkInterface(DeclaratorInterface[BackLinkProtodescriptor]):
    """Interface for backlink protodescriptor constructors."""
    __handle__ = "backlink"

    def __call__(
        self,
        *,
        via: type | Missing = MISSING,
        limit: int | Missing = MISSING,
        load: Literal["eager", "lazy"] | Missing = MISSING,
        repr: bool | Callable | str | Missing = MISSING,
        doc: str | Missing = MISSING,
        config: Mapping[str, Any] | Missing = MISSING,
    ) -> Any:
        """Construct a backlink protodescriptor with current AML field semantics."""
        return BackLinkProtodescriptor(
            via=via,
            limit=limit,
            load=load,
            repr=repr,
            doc=doc,
            config=config,
        )


class ParserInterface(DeclaratorInterface[ParserDeclarator]):
    """Interface for parser declarators."""
    __handle__ = "parser"

    def __call__(self, *members: str) -> Callable[[Callable[[Arche | prototype, Any], Any]], ParserDeclarator]:  # noqa
        """Create a parser declarator as a decorator."""
        def decorator(fn: Callable[[Arche | prototype, Any], Any]) -> ParserDeclarator:
            doc = MISSING if fn.__doc__ is None else fn.__doc__
            declarator = ParserDeclarator(callable=fn, members=members, doc=doc)
            update_wrapper(declarator, fn, updated=())  # type: ignore[arg-type]
            return declarator

        return decorator


class ValidatorInterface(DeclaratorInterface[ValidatorDeclarator]):
    """Interface for validator declarators."""
    __handle__ = "validator"

    def __call__(self, *members: str) -> Callable[[Callable[[Arche | prototype, Any], bool]], ValidatorDeclarator]:  # noqa
        """Create a validator declarator as a decorator."""
        def decorator(fn: Callable[[Arche | prototype, Any], bool]) -> ValidatorDeclarator:
            doc = MISSING if fn.__doc__ is None else fn.__doc__
            declarator = ValidatorDeclarator(callable=fn, members=members, doc=doc)
            update_wrapper(declarator, fn, updated=())  # type: ignore[arg-type]
            return declarator

        return decorator

# endregion

# =============================================================================
# Singleton instances
# =============================================================================
# region Singleton instances


metadata = MetadataInterface()
option = OptionInterface()
nxfield = NxFieldInterface()
data = DataInterface()
link = LinkInterface()
backlink = BacklinkInterface()
parser = ParserInterface()
validator = ValidatorInterface()
meta = metadata

# endregion
