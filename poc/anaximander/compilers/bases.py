from pathlib import Path
from types import ModuleType

from jinja2 import Environment, PackageLoader, Template

from .. import Project
from ..aml import Metadescriptor, Model
from ..utils.funcs import subclasses

J2ENV = Environment(
    loader=PackageLoader("anaximander.compilers"), trim_blocks=True, lstrip_blocks=True
)


def get_macro(template, name):
    macro = template._TemplateReference__context.vars[name]
    if callable(macro):
        return macro
    raise ValueError(f"Macro '{name}' not found.")


J2ENV.globals["get_macro"] = get_macro

for meatadescriptor_type in subclasses(Metadescriptor, strict=False):
    J2ENV.globals[meatadescriptor_type.__name__] = meatadescriptor_type


class ModuleCompiler:
    """Encapsulates the steps to compile a module."""

    __types__ = {}

    def __init_subclass__(cls, handle: str):
        assert isinstance(handle, str)
        assert handle not in cls.__types__
        cls.handle = handle
        cls.__types__[handle] = cls

    def __class_getitem__(cls, key):
        return cls.__types__[key]

    def __init__(self, module: ModuleType, destination: Path | str | None = None):
        self.module = module
        self.destination = Path(destination) if destination else None

    @property
    def template(self) -> Template:
        template_name = f"{self.handle}/{self.handle}.py.j2"
        return J2ENV.get_template(template_name)

    @property
    def modeltypes(self) -> list[type[Model]]:
        rval = []
        for k, v in self.module.__dict__.items():
            if isinstance(v, type):
                if issubclass(v, Model):
                    if self.handle in v.__compilations__:
                        rval.append(v)
        return rval

    def __call__(self, **kwargs):
        code = self.template.render(compiler=self, **kwargs)
        if write_path := self.destination:
            write_path.parent.mkdir(parents=True, exist_ok=True)
            with open(write_path, "w") as f:
                f.write(code)
        return code


class ProjectCompiler:
    """Encapsulates the steps to compile a project."""

    def __init__(self, project: Project, *compilations: str):
        self.project = project
        self.compilations = list(compilations) or list(ModuleCompiler.__types__)

    def __call__(self, **kwargs):
        modules = self.project.import_nxmodels_modules()
        models_path = self.project.models_path
        compile_path = self.project.compile_path
        for handle in self.compilations:
            compiler_class = ModuleCompiler[handle]
            for module in modules:
                if not (spec := module.__spec__):
                    raise RuntimeError(f"Module {module} has no spec.")
                if not (origin := spec.origin):
                    raise RuntimeError(f"Module {module} has no origin.")
                module_path = Path(origin)
                relative_path = module_path.relative_to(models_path)
                destination = (compile_path / f"{handle}_" / relative_path).as_posix()
                module_compiler = compiler_class(module, destination=destination)
                module_compiler(**kwargs)


# Monkey patching Project to include a compile method.
def _compile_project(self: Project, *compilations: str, **kwargs):
    compiler = ProjectCompiler(self, *compilations)
    compiler(**kwargs)


Project.compile = _compile_project
