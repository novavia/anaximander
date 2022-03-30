from typing import Any
from anaximander.meta import nxobject, nxtype, archetype
from anaximander.descriptors import metadata


def myfunction(name):
    return "Hello " + name


@archetype
class Object(nxobject, metaclass=nxtype):
    param: str = metadata(typespec=True)
