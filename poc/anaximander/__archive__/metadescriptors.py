from abc import ABC, abstractmethod

from types import NoneType, UnionType
from typing import Callable, ClassVar, Union, get_args, get_origin


import attrs


@attrs.define
class metadescriptor(ABC):
    __reserved_names__: ClassVar[list[str]] = []
    name: str = attrs.field(init=False)

    def __init_subclass__(cls):
        super().__init_subclass__()
        parent = cls.mro()[0]
        reserved_names: list[str] = getattr(parent, "__reserved_names__", [])
        if "__reserved_names__" in vars(cls):
            cls.__reserved_names__ = reserved_names + cls.__reserved_names__
        else:
            cls.__reserved_names__ = reserved_names

    def __set_name__(self, owner: type, name: str):
        if name in self.__reserved_names__:
            msg = f"Cannot use reserved name {name} for {self.__class__.__name__} descriptor."
            raise ValueError(msg)
        self.name = name


@attrs.define
class annotated_metadescriptor(metadescriptor):
    annotation: str | None = attrs.field(init=False)

    @abstractmethod
    def validate_annotation(self, annotation):
        return NotImplemented

    def __set_name__(self, owner: type, name: str):
        super().__set_name__(owner, name)
        self.annotation = getattr(owner, "__annotations__", {}).get(name)

    # @classmethod
    # def hint_compiler(cls, hint) -> str:
    #     if hint in (None, NoneType):
    #         return "None"
    #     elif isinstance(hint, type):
    #         return hint.__name__
    #     origin = get_origin(hint)
    #     args = get_args(hint)
    #     match origin:
    #         case UnionType():
    #             return " | ".join(cls.hint_compiler(arg) for arg in args)


    # @property
    # def hint(self):
    #     return self.hint_compiler(self.annotation)
