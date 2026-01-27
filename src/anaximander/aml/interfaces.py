"""Interface objects for AML declarators and metadata bindings."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


from collections.abc import Mapping
from functools import update_wrapper
from numbers import Real
from typing import Any, Callable, Literal

from anaximander.utils.meta import Singleton

from .declarative import (
    DECLARATIVE_NAMESPACE,
    MISSING,
    DeclarativeNamespace,
    Declarator,
    EnumerationCallableDeclarator,
    Missing,
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
from .protodescriptors import BackLinkProtodescriptor, DataProtodescriptor, LinkProtodescriptor

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
    __validator_type__: type[EnumerationCallableDeclarator] | None = None

    def __init_subclass__(cls) -> None:
        try:
            if "handle" in cls.__dict__:
                cls.__declarator_type__ = Declarator.__handles__[cls.__handle__]
        except KeyError:
            raise ValueError(f"Invalid handle '{cls.__handle__}' for declarator interface.")

    @property
    def declarator_type(self) -> type[DT]:
        """The declarator type for this interface."""
        return self.__class__.__declarator_type__

    @property
    def validator_type(self) -> type[EnumerationCallableDeclarator] | None:
        """The validator declarator type for this interface, if any."""
        return self.__validator_type__

    @property
    def handle(self) -> str:
        """The interface handle."""
        return self.__class__.__handle__

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__") or name in dir(self):
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

    def validator(self, *members: str) -> Callable[[Callable], Declarator]:
        """Create a validation declarator as a decorator."""
        if (validator_type := self.validator_type) is None:
            raise TypeError(f"{self.handle} interface does not support validators.")

        def decorator(fn: Callable) -> Declarator:
            declarator = validator_type(callable=fn, members=members, doc=fn.__doc__)
            update_wrapper(declarator, fn, updated=())  # type: ignore[arg-type]
            return declarator

        return decorator


class AssignableMetadescriptorInterface[DT](DeclaratorInterface[AssignableMetadescriptor]):
    """Interface for metadescriptor declarators and bindings."""

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
    ) -> MetadataDeclarator:
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


class DataInterface(DeclaratorInterface[DataProtodescriptor]):
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
        config: Mapping[str, Any] | None | Missing = MISSING,
    ) -> DataProtodescriptor:
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


class LinkInterface(DeclaratorInterface[LinkProtodescriptor]):
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
        config: Mapping[str, Any] | None | Missing = MISSING,
    ) -> LinkProtodescriptor:
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
        config: Mapping[str, Any] | None | Missing = MISSING,
    ) -> BackLinkProtodescriptor:
        """Construct a backlink protodescriptor with current AML field semantics."""
        return BackLinkProtodescriptor(
            via=via,
            limit=limit,
            load=load,
            repr=repr,
            doc=doc,
            config=config,
        )

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
meta = metadata

# endregion
