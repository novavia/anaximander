from importlib import import_module
from pathlib import Path
from types import ModuleType
import importlib.util
import sys

from jinja2 import Environment, PackageLoader, Template

from ..aml.modeltype import modeltype

J2ENV = Environment(
    loader=PackageLoader("anaximander.compilers"), trim_blocks=True, lstrip_blocks=True
)

def import_from_path(module_name: str, file_path: Path | str, package: bool = False):
    """Programatically imports a module from its name and file path."""
    # if package:
    sys.path.append(file_path.parent.as_posix())
    module_or_package = import_module(file_path.stem)
    sys.path.remove(file_path.parent.as_posix())
    return module_or_package
    # spec = importlib.util.spec_from_file_location(module_name, file_path)
    # module = importlib.util.module_from_spec(spec)
    # sys.modules[module_name] = module
    # spec.loader.exec_module(module)
    # return module


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

    def __init__(self, module: ModuleType, *, destination: Path | str = None):
        self.module = module
        self.destination = Path(destination) if destination else None

    @classmethod
    def from_path(cls, path: Path | str, destination: Path | str = None):
        module = import_from_path(path.stem, path)
        return cls(module, destination)

    @property
    def template(self) -> Template:
        template_name = f"{self.handle}/{self.handle}.py.j2"
        return J2ENV.get_template(template_name)

    @property
    def source_path(self) -> Path:
        return Path(self.module.__spec__.origin)

    @property
    def destination_path(self) -> Path:
        if self.destination is None:
            return None
        return (self.destination / self.module.__name__).with_suffix(".py")

    @property
    def modeltypes(self) -> list[modeltype]:
        rval = []
        for k, v in self.module.__dict__.items():
            if isinstance(v, modeltype):
                if self.handle in v.__compilations__:
                    rval.append(v)
        return rval

    def __call__(self, **kwargs):
        code = self.template.render(compiler=self, **kwargs)
        if write_path := self.destination_path:
            write_path.parent.mkdir(parents=True, exist_ok=True)
            with open(write_path, "w") as f:
                f.write(code)
        return code


class PackageCompiler:

    def __init__(self, package: ModuleType, *compilations: str, destination: str = None):
        self.package = package
        self.compilations = list(compilations) or list(ModuleCompiler.__types__)
        self.destination = destination

    @classmethod
    def from_path(cls, path: Path | str, *compilations: str, destination: Path | str = None):
        sys.path.append(path.parent.as_posix())
        import_module(path.stem)
        package = sys.modules[path.stem]
        sys.path.remove(path.parent.as_posix())
        return cls(package, *compilations, destination=destination)

    @property
    def source_path(self):
        return Path(self.package.__path__[-1])

    def modules(self, recursive: bool = False):
        if recursive:
            module_paths = self.source_path.glob("**/*.py")
        else:
            module_paths = self.source_path.glob("*.py")
        return [import_from_path(mp.stem, mp) for mp in module_paths]

    def __call__(self, **kwargs):
        for handle in self.compilations:
            module_compiler_type = ModuleCompiler[handle]
            compile_path = self.destination.format(handle = (handle + "_"))
            for module in self.modules():
                module_compiler = module_compiler_type(module, destination=compile_path)
                module_compiler(**kwargs)


class ProjectCompiler:

    def __init__(self, path: Path | str, *compilations: str):
        self.path = Path(path)
        self.compilations = list(compilations) or list(ModuleCompiler.__types__)

    @property
    def project_name(self):
        return self.path.name

    def __call__(self, **kwargs):
        models_path = self.path / "src/nxmodels"
        compile_path = self.path / f"src/{self.project_name}"
        if not models_path.exists():
            msg = f"Project {self.path.name} does not contain the requisite nxmodels folder."
            raise FileNotFoundError(msg)
        packages = set(p.parent for p in models_path.rglob("*.py"))
        packages = list(sorted(packages, key=lambda p: len(p.parts), reverse=True))
        for package in packages:
            destination = (compile_path / "{handle}" / package.relative_to(models_path)).as_posix()
            package_compiler = PackageCompiler.from_path(package,
                                                         *self.compilations,
                                                         destination=destination
                                                         )
            package_compiler(**kwargs)
