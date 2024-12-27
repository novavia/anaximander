from functools import wraps

from .metadescriptors import metadescriptor, field
from .modeltype import Model


def compile(*compilers: str, **kwargs):
    """A class decorator factory that flags a model for compilations."""
    def decorator(cls: type[Model]):
        for handle in compilers:
            cls.__compilations__[handle] = dict(kwargs)
        return cls
    return decorator
