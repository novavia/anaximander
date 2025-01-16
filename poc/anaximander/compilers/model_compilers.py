from .bases import ModuleCompiler


class DataclassCompiler(ModuleCompiler, handle="dataclasses"):
    pass


class PydanticCompiler(ModuleCompiler, handle="pydantic"):
    pass


class SQLAlchemyCompiler(ModuleCompiler, handle="sqlalchemy"):
    pass
