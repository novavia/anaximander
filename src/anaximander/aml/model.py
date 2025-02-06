from typing import dataclass_transform

from .meta import DataobjectMetadescriptor, ModelABC, Prototype
from .modeldescriptors import field, parent, query


class ModelType(Prototype):
    """Metaclass for model classes."""

    pass


@dataclass_transform(field_specifiers=(DataobjectMetadescriptor, field, parent, query))
class Model(ModelABC, metaclass=ModelType):
    pass
