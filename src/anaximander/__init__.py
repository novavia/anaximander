from .projects import NxModuleType, Project, NXPATH
from .aml import (
    Data,
    Metadescriptor,
    Model,
    Prototype,
    compile,
    data,
    dataobject,
    field,
    model,
    parent,
    prototype,
    query,
)
from .compilers import ProjectCompiler

REPO = NXPATH.parent.parent

__all__ = [
    "NxModuleType",
    "Project",
    "NXPATH",
    "REPO",
    "Data",
    "Metadescriptor",
    "Model",
    "Prototype",
    "compile",
    "data",
    "dataobject",
    "field",
    "model",
    "parent",
    "prototype",
    "query",
    "ProjectCompiler",
]
