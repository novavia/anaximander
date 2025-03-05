from functools import singledispatchmethod
from typing import get_origin

from ..aml.meta import Metadescriptor
from ..aml.modeldescriptors import Field, Parent, Query
from .bases import ModuleCompiler


class DataclassCompiler(ModuleCompiler, handle="dataclasses"):
    @singledispatchmethod
    def descriptor(self, metadescriptor: Metadescriptor) -> str:
        return super().descriptor(metadescriptor)

    @descriptor.register
    def field_descriptor(self, field: Field) -> str:
        name = field.name
        hint = field.hint
        if isinstance(hint, type):
            field_type = hint
        else:
            field_type = get_origin(hint)
        annotation = field_type.__name__
        assignment = "field()"
        return self._print_descriptor(name, annotation, assignment)


class PydanticCompiler(ModuleCompiler, handle="pydantic"):
    pass


class SQLAlchemyCompiler(ModuleCompiler, handle="sqlalchemy"):
    @singledispatchmethod
    def descriptor(self, metadescriptor: Metadescriptor) -> str:
        return super().descriptor(metadescriptor)

    @descriptor.register
    def field_descriptor(self, field: Field) -> str:
        name = field.name
        hint = field.hint
        if isinstance(hint, type):
            field_type = hint
        else:
            field_type = get_origin(hint)
        annotation = f"Mapped[{field_type.__name__}]"
        assignment = "mapped_column()"
        return self._print_descriptor(name, annotation, assignment)

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
