from typing import Callable

import attrs

from . import dataobject_metadescriptor


@attrs.define
class field(dataobject_metadescriptor):
    key: bool | str | list[str] | None = attrs.field(default=None)
    sequence: bool | str | list[str] | None = attrs.field(default=None)
    group: str | list[str] | None = attrs.field(default=None)
    index: bool | str | list[str] | None = attrs.field(default=None)
    unique: bool | None = attrs.field(default=None)
    repr: bool | Callable | None = attrs.field(default=None)

    def validate_hint(self, annotation):
        return super().validate_hint(annotation)


@attrs.define
class relationship(dataobject_metadescriptor):
    pass
