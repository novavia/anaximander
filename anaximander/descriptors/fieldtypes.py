"""This module sets base classes for admissible metadata types."""

from collections import deque
import enum
from types import GenericAlias
import typing
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Union,
    _Final,
)

from beartype._decor.main import beartype

from ..utilities.functions import singleton


Hint = Union[GenericAlias, _Final]
NoneType = type(None)
HintObject = (GenericAlias, _Final)
function = type(lambda: True)


@singleton
class UNDEFINED:
    """A singleton token for distinguishing undefined default values from None."""

    pass


class nxfieldtype(type):
    """The metaclass for nxfield."""

    __collections__: ClassVar[set[type]] = {
        list,
        tuple,
        dict,
        set,
        frozenset,
        deque,
    }
    __typing_collections__: ClassVar[dict[type, type]] = {
        typing.List: list,
        typing.Tuple: tuple,
        typing.Dict: dict,
        typing.Set: set,
        typing.FrozenSet: frozenset,
        typing.Deque: deque,
    }
    __collections_typing__: ClassVar[dict[type, type]] = {
        list: typing.List,
        tuple: typing.Tuple,
        dict: typing.Dict,
        set: typing.Set,
        frozenset: typing.FrozenSet,
        deque: typing.Deque,
    }

    def __instancecheck__(cls, object_: Any) -> bool:
        try:
            type_ = type(object_)
            if cls.from_hint(type_) == type_:
                return True
            return super().__subclasscheck__(cls, type_)
        except TypeError:
            return False

    def __subclasscheck__(cls, type_: type) -> bool:
        try:
            if cls.from_hint(type_) == type_:
                return True
            return super().__subclasscheck__(cls, type_)
        except TypeError:
            return False

    @classmethod
    def from_hint(mcl, hint: Union[type, Hint]) -> type["nxfield"]:
        """This function normalizes and validates hints as nxfield."""
        if hint is Any:
            return object
        elif isinstance(hint, type):
            return hint
        elif isinstance(hint, enum.EnumMeta):
            values = [m._value_ for m in hint.__members__.values()]
            # All values must be of the same field type
            types = set(type(v) for v in values)
            if len(types) == 1:
                return types.pop()
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)
        if origin in mcl.__typing_collections__:
            return mcl.__typing_collections__[origin]
        elif origin in mcl.__collections__:
            # We validate the hint because there is no enforcement
            # when using built-in types
            typing_origin = mcl.__collections_typing__[origin]
            # This will raise TypeError if the hint is wrongly formulated
            typing_origin[args]
            return origin
        elif origin is Union:  # type: ignore
            types = set(args)
            if NoneType in types:
                types -= {NoneType}
                if len(types) == 1:
                    return types.pop()
        elif origin is Literal:
            # All arguments must be of the same field type
            types = set(type(a) for a in typing.get_args(hint))
            if len(types) == 1:
                return types.pop()
        elif origin is Callable:
            return function
        msg = f"Unsupported type hint {hint}"
        raise TypeError(msg)

    def __call__(
        cls,
        data,
        *,
        spec: Optional[Hint] = None,
    ):
        """Redefines call for nxfield, which is abstract and doesn't produce instances."""
        if spec is None:
            return data

        nxfield_type = cls.from_hint(spec)

        @beartype
        def typecheck(metadata: nxfield_type):
            return True

        try:
            assert typecheck(data)
        except:
            msg = f"Incorrect data {data} supplied for {spec}"
            raise TypeError(msg)
        else:
            return data


class nxfield(type, metaclass=nxfieldtype):
    """An abstract base class for all field types."""

    pass


FieldType = type[nxfield]


def field_type_from_hint(hint: Union[type, Hint]) -> FieldType:
    return nxfield.from_hint(hint)
