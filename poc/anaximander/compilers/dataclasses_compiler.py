from . import ModuleCompiler


class DataclassCompiler(ModuleCompiler, handle="dataclasses"):
    pass


class PydanticCompiler(ModuleCompiler, handle="pydantic"):
    pass
