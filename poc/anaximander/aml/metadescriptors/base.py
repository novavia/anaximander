from abc import ABC, abstractmethod
from types import GenericAlias, UnionType
from typing import Any, ClassVar, get_type_hints

import attrs

from .datatypes import dataobject


@attrs.define
class metadescriptor(ABC):
    __reserved_names__: ClassVar[set[str]] = set()
    name: str = attrs.field(init=False)

    def __init_subclass__(cls):
        super().__init_subclass__()
        parent = cls.mro()[1]
        reserved_names: set[str] = getattr(parent, "__reserved_names__", set())
        if "__reserved_names__" in vars(cls):
            cls.__reserved_names__ = reserved_names | set(cls.__reserved_names__)
        else:
            cls.__reserved_names__ = reserved_names

    def __set_name__(self, owner: type, name: str):
        if name in self.__reserved_names__:
            mdtype = self.__class__.__name__
            msg = (
                f"Cannot use reserved name {name} for metadescriptor or type {mdtype}."
            )
            raise ValueError(msg)
        self.name = name


@attrs.define
class typed_metadescriptor(metadescriptor):
    hint: Any = attrs.field(init=False)

    @abstractmethod
    def validate_hint(self, hint: Any) -> bool:
        return True

    def __set_name__(self, owner: type, name: str):
        super().__set_name__(owner, name)
        mdtype = self.__class__.__name__
        if "__type_hints__" not in vars(owner):
            setattr(owner, "__type_hints__", get_type_hints(owner))
        hint = getattr(owner, "__type_hints__", {}).get(name)
        # Case no hint
        if hint is None:
            msg = (
                f"Metadescriptor {name} of class {mdtype} must be supplied a type hint."
            )
        if not self.validate_hint(hint):
            msg = f"Invalid type annotation {hint} supplied to metadescriptor {name} of class {mdtype}."
            raise TypeError(msg)
        self.hint = hint


@attrs.define
class dataobject_metadescriptor(typed_metadescriptor):
    hint: type[dataobject] | GenericAlias | UnionType = attrs.field(init=False)

    def validate_hint(self, hint: Any) -> bool:
        return dataobject.validate_hint(hint)
