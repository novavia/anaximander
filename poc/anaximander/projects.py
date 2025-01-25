import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

import attrs

NXPATH = Path(__file__).parent


@attrs.define
class Project:
    """A class that represents an Anaximander project."""

    path: Path = attrs.field(converter=Path)

    @property
    def name(self):
        return self.path.name

    @property
    def models_path(self) -> Path:
        return self.path / "src/nxmodels"

    @property
    def compile_path(self) -> Path:
        return self.path / f"src/{self.name}"

    @classmethod
    def _import_from_directory(
        cls, directory: Path, package: str | None = None
    ) -> list[ModuleType]:
        """Primitive function for import_models.

        Args:
            directory (Path): The directory containing python files to import.
            package (str | None, optional): Optional parent package, possibly nested.
                Defaults to None.

        Returns:
            list[ModuleType]: A list of imported modules.
        """
        modules = []
        subdirectories = directory.glob("*/")
        for sub in subdirectories:
            if sub.name == "__pycache__":
                continue
            subpackage = sub.name if package is None else f"{package}.{sub.name}"
            modules.extend(cls._import_from_directory(sub, package=subpackage))
        module_paths = directory.glob("*.py")
        for module_path in module_paths:
            module_name = module_path.stem
            if package is not None:
                module_name = "." + module_name
            modules.append(import_module(module_name, package=package))
        return modules

    def import_nxmodels_modules(self) -> list[ModuleType]:
        """Imports the modules in the project's src/nxmodels directory."""
        if not self.models_path.exists():
            msg = f"Project {self.name} does not contain the requisite nxmodels folder."
            raise FileNotFoundError(msg)
        if self.models_path not in sys.path:
            sys.path.insert(0, self.models_path.as_posix())
        return self._import_from_directory(self.models_path)

    def compile(self, *compilations: str, **kwargs):
        """Compiles the project using the specified compilers."""
        return None
