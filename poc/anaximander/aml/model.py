from typing import dataclass_transform

from .meta import ModelABC, Prototype
from .modeldescriptors import Field, field


class ModelType(Prototype):
    """Metaclass for model classes."""

    pass


@dataclass_transform(field_specifiers=(Field, field))
class Model(ModelABC, metaclass=ModelType):
    pass
