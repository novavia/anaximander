from typing import Callable, TypeVar, dataclass_transform

from ..metadescriptors import Field, field, model
from .base import Prototype


class ModelType(Prototype):
    """Metaclass for model classes."""

    pass


@dataclass_transform(field_specifiers=(Field, field))
class Model(model, metaclass=ModelType):
    pass


M = TypeVar("M", bound=Model)


def compile(*compilers: str, **kwargs) -> Callable[[type[M]], type[M]]:
    """A class decorator factory that flags a model for compilations."""

    def decorator(cls: type[M]) -> type[M]:
        for handle in compilers:
            cls.__compilations__[handle] = dict(kwargs)
        return cls

    return decorator
