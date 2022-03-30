"""This module defines descriptors that wrap methods."""

from dataclasses import dataclass
import dataclasses as dc
from functools import wraps
import inspect
from typing import Callable


from ..utilities.functions import (
    mandatory_arguments,
    named_arguments,
    takes_keyword_arguments,
    typeproperty,
)
from ..utilities.nxdataclasses import Validated

from .base import MetaDescriptor


class MetaMethod(MetaDescriptor, Validated):
    """Base class for Metaproperty and Metamethod.

    These descriptors are declared by decorating methods in archetype
    declarations.
    Metaproperties are written as static methods that take metadata as their
    only arguments.
    By contrast, metamethods are written as class methods, and their signature
    can also be supplemented with metadata arguments for clarity (technically,
    these same metadata arguments are attributes of the calling class).
    The metadata arguements may be named arguments, positional or keyword-only,
    or as a catch-all **kwargs or **metadata.
    The decorated metamethods are run by metaclasses at class creation.
    As the names indicate, metaproperties return arbitary properties of the
    metadata, whereas metamethods are written as closures and return a
    method implementation for the class being created.
    """

    # Init fields
    name: str
    method: Callable = dc.field(repr=False)

    # Automatic fields
    method_args: list[str] = dc.field(repr=False)  # Holds method arguments
    mandatory_args: list[str] = dc.field(repr=False)  # Holds mandatory arguments
    keyword_args: bool = dc.field(repr=False)  # Whether the method accepts **kwargs

    @classmethod
    def validate_call_signature(cls, method):
        """Checks the method signature to avoid deeper bugs down the line.

        Admissible signatures:
            @metaproperty
            def cool(**metadata): ...
            @metaproperty
            def cool(**kwargs): ...
            @metaproperty
            def cool(x, y=0, **kwargs): ...
            @metaproperty
            def scalable(scale): ...

        Note that adding **kwargs is optional and safe to do
        even if some metadata attributes are omitted from the call signature,
        because the eventual call to the method will intercept those
        superfluous arguments.
        """
        no_args = False
        signature = inspect.signature(method)
        for param in signature.parameters.values():
            if param.kind is param.VAR_POSITIONAL:
                break
        else:
            no_args = True
        return no_args


@dataclass
class Metaproperty(MetaMethod):
    """A property of metadata, run by a metaclass prior to type creation.

    Metaproperties can be thought of as properties of types to the extent
    that metadata are the attributes of types.
    However, these are defined without reference to a type -i.e. their
    signature is prop(meta, **metadata), where meta is the metaclass. The
    reason for this is that metaproperties can be evaluated before an
    object instance is even created (i.e. in the caller class' __new__ method),
    and in this way modify the type of object that ends up being generated -hence
    the name 'metaproperty', since these can point to a different construction
    method.
    The mechanism for this is metacharacters. A metacharacter provides an
    interface between declared metaproperties and traits that are included
    in an object's type. Effectively, the metaproperty decorator accepts
    any number of strings that are treated as metacharacters. If a metaproperty's
    return value evaluates to True, then the metacharacters are conferred to
    the caller. Note that the metaproperty's return value need not be a boolean.
    Any object can be returned, but it will be submitted to an assertion test
    to determine whether or not the metaproperty is carried.
    """

    handle = "metaproperty"
    handle_alternatives = ["metaproperties"]

    # Init fields
    name: str
    method: Callable = dc.field(repr=False)
    metacharacters: list[str] = dc.field()

    # Automatic fields
    method_args: list[str] = dc.field(repr=False)  # Holds method arguments
    mandatory_args: list[str] = dc.field(repr=False)  # Holds mandatory arguments
    keyword_args: bool = dc.field(repr=False)  # Whether the method accepts **kwargs

    def __init__(self, method: Callable, *metacharacters: str):
        self.metacharacters = list(metacharacters)
        self.name = method.__name__
        self.method = method
        if not self.validate_call_signature(method):
            msg = "Incorrect call signature passed to metaproperty"
            raise TypeError(msg)
        self.method_args = named_arguments(self.method)
        self.mandatory_args = mandatory_arguments(self.method)
        self.keyword_args = takes_keyword_arguments(self.method)
        Validated.__post_init__(self)

    def __call__(self, **metadata):
        """Calls the method against the supplied metadata."""
        if not all(m in metadata for m in self.mandatory_args):
            return None
        if self.keyword_args:
            kwargs = metadata
        else:
            kwargs = {k: v for k, v in metadata.items() if k in self.method_args}
        try:
            return self.method(**kwargs)
        except:  # This is opaque, but primarily intended to deal with None values
            return None

    def _make_type_property(self):
        @wraps(self.method)
        def wrapped(cls):
            return self(**cls.metadata)

        return typeproperty(wrapped)

    def set_attribute(self, host: type):
        prop = self._make_type_property()
        setattr(host, self.name, prop)


@dataclass
class Metamethod(MetaMethod):
    """A closure mechanism to specify type methods in archetypes.

    The rationale for metamethods is to utilize the attributes of a type
    in order to produce a more efficient runtime method. Most typically,
    this would include data construction methods that must compose an
    archetype and a data model, and are hence best computed at type
    initialization.
    Unike metaproperties, metamethods' signature must be of the form
    (cls, arg1, *, arg2, **kwargs) where named arguments are all
    declared metadata.
    The metaclass runs the metamethods at type initialization and expects
    a method in return. Hence metamethods are written as closures, that is,
    their return value is a method declaration, and that method declaration
    gets appended to new type, though with its name changed to that of the
    declared metamethod, as so:

    @metamethod
    def validate(cls, dataschema, **metadata):
        validator = dataschema.validator(**metadata)
        def validation_method(cls_, data):
            return validator(data)
        return validation_method

    Subtypes of an archetype that declares the above metamethod will have
    a 'validate' method that runs validation_method. In many cases,
    using closures can provide a performance gain in being a form of
    pre-compilation. In the above example, the schema interface is called once
    and for all when the type is created.
    The return method can be a classmethod or staticmethod as well, and
    will be passed on as such to subtypes.
    """

    handle = "metamethod"
    handle_alternatives = ["metamethods"]

    # Init fields
    name: str
    method: Callable[..., Callable] = dc.field(repr=False)

    # Automatic fields
    method_args: list[str] = dc.field(repr=False)  # Holds method arguments
    mandatory_args: list[str] = dc.field(repr=False)  # Holds mandatory arguments
    keyword_args: bool = dc.field(repr=False)  # Whether the method accepts **kwargs

    def __init__(self, method: Callable[..., Callable]):
        self.name = method.__name__
        self.method = method
        if not self.validate_call_signature(method):
            msg = "Incorrect call signature passed to metaproperty"
            raise TypeError(msg)
        # Note that the first mandatory argument is 'cls', which is removed
        self.method_args = named_arguments(self.method)[1:]
        self.mandatory_args = mandatory_arguments(self.method)[1:]
        self.keyword_args = takes_keyword_arguments(self.method)
        Validated.__post_init__(self)

    @classmethod
    def validate_call_signature(cls, method):
        """Checks the method signature to avoid deeper bugs down the line.

        Admissible signatures:
            @metamethod
            def cool(cls, **metadata): ...
            @metamethod
            def cool(cls, **kwargs): ...
            @metamethod
            def cool(cls, x, y=0, **kwargs): ...
            @metamethod
            def scalable(cls, scale): ...

        To sum up, a positional argument 'cls', and no star
        arguments. Note that adding **kwargs is optional and safe to do
        even if some metadata attributes are omitted from the call signature,
        because the eventual call to the method will intercept those
        superfluous arguments.
        """
        class_arg = False
        no_args = False
        signature = inspect.signature(method)
        for param in signature.parameters.values():
            if param.name == "cls":
                if param.kind is param.POSITIONAL_OR_KEYWORD:
                    class_arg = True
            if param.kind is param.VAR_POSITIONAL:
                break
        else:
            no_args = True
        return class_arg and no_args

    def __call__(self, host, **metadata):
        """Calls the method against the supplied host and metadata."""
        if not all(m in metadata for m in self.mandatory_args):
            return None
        if self.keyword_args:
            kwargs = metadata
        else:
            kwargs = {k: v for k, v in metadata.items() if k in self.method_args}
        try:
            return self.method(host, **kwargs)
        except:
            # Metamethods are generally not computable on abstract classes
            # because by definition they lack metadata. For that reason,
            # the method is simply set to None.
            if getattr(host, "abstract", True):
                return None
            else:
                raise

    def set_attribute(self, host: type):
        """Sets the target method on host."""
        metadata = getattr(host, "metadata", {})
        method = self(host, **metadata)
        setattr(host, self.name, method)
