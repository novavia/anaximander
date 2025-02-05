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

    @property
    def app_models_path(self) -> Path:
        return self.compile_path / "nxmodels_"

    @classmethod
    def _collect_from_directory(cls, directory: Path) -> list[Path]:
        """Primitive method for collect_nxmodels_modules.

        Args:
            directory (Path): The directory containing python files to import.

        Returns:
            list[Path]: A list of module file paths.
        """
        module_paths = []
        subdirectories = directory.glob("*/")
        for sub in subdirectories:
            if sub.name == "__pycache__":
                continue
            module_paths.extend(cls._collect_from_directory(sub))
        module_paths.extend(directory.glob("*.py"))
        return module_paths

    def collect_nxmodels_modules(self) -> list[Path]:
        """Collects modules from the nxmodels directory."""
        if not self.models_path.exists():
            msg = f"Project {self.name} does not contain the requisite nxmodels folder."
            raise FileNotFoundError(msg)
        return self._collect_from_directory(self.models_path)

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
        """Imports the modules in the project's src/<project>/api/nxmodels_ directory."""
        if not self.app_models_path.exists():
            msg = f"Project {self.name} does not contain the requisite nxmodels_ folder."
            raise FileNotFoundError(msg)
        if self.app_models_path not in sys.path:
            sys.path.insert(0, self.app_models_path.as_posix())
        modules = self._import_from_directory(self.app_models_path)
        # Next we validate that no import targets the original modules in the nxmodels
        # directory, which could be the case if absolute imports are used
        for name, module in sys.modules.items():
            try:
                module_path = Path(module.__file__)  # type: ignore
            except (AttributeError, TypeError, ValueError):
                continue
            if module_path.is_relative_to(self.models_path):
                msg = (
                    "Modules in the nxmodels directory that import other modules in that "
                    + f"directory must use relative syntax. {name} in module "
                    + f"{module.__name__} does not."
                )
                raise ImportError(msg)
        return modules

    def copy_nxmodels_modules(self):
        """Copies modules from the src/nxmodels directory to the application directory."""
        models_path = self.models_path
        copy_path = self.app_models_path
        for module_path in self.collect_nxmodels_modules():
            relative_path = module_path.relative_to(models_path)
            destination = copy_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            module_path.copy(destination)

    def compile(self, *compilations: str, **kwargs):
        """Compiles the project using the specified compilers."""
        return None
