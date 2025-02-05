import typing
from enum import Enum
from functools import singledispatchmethod
from types import NoneType, UnionType
from typing import Any, Literal

from ..aml.meta import DataObjectABC, Metadescriptor, data
from ..aml.modeldescriptors import Field, Parent, Query
from .bases import ModuleCompiler


class DataclassCompiler(ModuleCompiler, handle="dataclasses"):
    pass


class PydanticCompiler(ModuleCompiler, handle="pydantic"):
    pass


class SQLAlchemyCompiler(ModuleCompiler, handle="sqlalchemy"):
    def validate_field_hint(self, hint: Any) -> bool:
        if isinstance(hint, type):
            if issubclass(hint, Enum):
                values = [m._value_ for m in hint.__members__.values()]
                vtypes = {type(v) for v in values}
                return all(self.validate_field_hint(t) for t in vtypes)
            return issubclass(hint, data.__primitives__)
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)
        if origin in DataObjectABC.__collections__:
            return all(self.validate_field_hint(arg) for arg in args)
        elif origin in (UnionType, typing.Union):
            args = set(args)
            args.discard(NoneType)
            return all(self.validate_field_hint(arg) for arg in args)
        elif origin is Literal:
            if len(args) == 1:
                return self.validate_field_hint(type(args[0]))
            args = set(args)
            args.discard(None)
            return all(self.validate_field_hint(type(arg)) for arg in args)
        return False

    # def resolve_field_hint(self, hint: Any) -> tuple[Any, bool]:
    #     """Returns the field's type and its nullability."""
    #     nullable = False
    #     if isinstance(hint, type):
    #         if issubclass(hint, Enum):
    #             values = {m._value_ for m in hint.__members__.values()}
    #             if None in values:
    #                 nullable = True
    #                 values.remove(None)
    #             vtypes = {type(v) for v in values}
    #             if len(vtypes) == 1:
    #                 hint = list(vtypes)[0]

    #             return all(self.validate_field_hint(t) for t in vtypes)
    #         return issubclass(hint, data.__primitives__)
    #     origin = typing.get_origin(hint)
    #     args = typing.get_args(hint)
    #     if origin in DataObjectABC.__collections__:
    #         return all(self.validate_field_hint(arg) for arg in args)
    #     elif origin in (UnionType, typing.Union):
    #         args = set(args)
    #         args.discard(NoneType)
    #         return all(self.validate_field_hint(arg) for arg in args)
    #     elif origin is Literal:
    #         if len(args) == 1:
    #             return self.validate_field_hint(type(args[0]))
    #         args = set(args)
    #         args.discard(None)
    #         return all(self.validate_field_hint(type(arg)) for arg in args)
    #     return False

    @singledispatchmethod
    def descriptor(self, metadescriptor: Metadescriptor) -> str:
        return super().descriptor(metadescriptor)

    @descriptor.register
    def field_descriptor(self, field: Field) -> str:
        name = field.name
        # if not self.validate_field_hint(hint := field.hint):
        #     msg = f"Cannot compiled {field} with type hint {hint}."
        #     raise TypeError(msg)
        type = f"Mapped[{field.annotation}]"
        assignment = "mapped_column()"
        return self._print_descriptor(name, type, assignment)

    @descriptor.register
    def parent_descriptor(self, parent: Parent) -> str:
        name = parent.name
        type = f"Mapped[{parent.annotation}]"
        assignment = "mapped_column()"
        return self._print_descriptor(name, type, assignment)

    @descriptor.register
    def query_descriptor(self, query: Query) -> str:
        name = query.name
        type = f"Mapped[{query.annotation}]"
        assignment = "relationship()"
        return self._print_descriptor(name, type, assignment)
