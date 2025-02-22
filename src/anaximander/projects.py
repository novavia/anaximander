import shutil
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

import attrs
from cookiecutter.main import cookiecutter

from .utils import Config
from .utils.funcs import workdir

NXPATH = Path(__file__).parent
PROJECT_TEMPLATE = NXPATH / "config/projects/project_template"


class ProjectConfig(Config):
    name: str

    def cookiecutter_context(self):
        return {
            "project_name": self.name,
        }


@attrs.define
class Project:
    """A class that represents an Anaximander project."""

    path: Path = attrs.field(converter=Path)

    @classmethod
    def create(
        cls,
        project_name: str,
        parent_directory: str | Path | None = None,
        *,
        prototypes: str | Path | None = None,
        **kwargs,
    ) -> "Project":
        """Creates a new Anaximander project into the specified directory.

        Args:
            project_name (str): A name for the project, which also sets the name of the project
                directory. It can be capitalized, but note that the name is converted to lowercase
                to define a top-level Python package.
            parent_directory (str | Path | None, optional): The directory in which to create
                the project. It can be absolute or relative to the current working directory.
                If None (the default), then it is set to the current working directory.
            prototypes (str | Path | None, optional): An optional path to either a python file
                or directory containing prototoype declarations. The content is then copied to
                'src/prototypes' in the project folder to form the basis of the project.
                Like parent_directory it can be either absolute or relative to the current
                working directory.
                Note that the name of the file or directory is ignored:
                - If the path points to a python file, it is turned to 'src/prototypes/__init__.py'
                - If the path points to a directored, its content is copied to 'src/prototoypes'
                Hence if the prototypes are defined in a module my_prototypes.py and the name
                needs to be preserved, one can place the module in a folder by itself and point
                to that folder.
                Defaults to None.

        Returns:
            Project: A new Anaximander project instance.
        """
        config = ProjectConfig(name=project_name, **kwargs)
        if parent_directory is None:
            parent_directory = Path.cwd()
        else:
            parent_directory = Path(parent_directory)
        cookiecutter_context = config.cookiecutter_context()
        with workdir(parent_directory):
            cookiecutter(
                PROJECT_TEMPLATE.as_posix(), no_input=True, extra_context=cookiecutter_context
            )
        project = cls(path=parent_directory / project_name)
        config.save(project.config_path)
        if prototypes is not None:
            prototypes = Path(prototypes)
            if not prototypes.exists():
                raise FileNotFoundError(f"Prototypes path {prototypes} does not exist.")
            if prototypes.is_dir():
                destination = project.prototypes_source_path
                shutil.copytree(prototypes, destination, dirs_exist_ok=True)
            else:
                destination = project.prototypes_source_path / "__init__.py"
                shutil.copy(prototypes, destination)
        return project

    @property
    def name(self):
        return self.path.name

    @property
    def config_path(self) -> Path:
        return self.path / ".nxp"

    @property
    def config(self) -> ProjectConfig:
        return ProjectConfig.load(self.config_path)

    @property
    def prototypes_source_path(self) -> Path:
        return self.path / "src/prototypes"

    @property
    def application_path(self) -> Path:
        return self.path / f"src/{self.name}"

    @property
    def api_prototypes_path(self) -> Path:
        return self.application_path / "api/prototypes_"

    @classmethod
    def is_project_directory(cls, path: Path | str) -> bool:
        return (Path(path) / ".nxp").exists()

    @classmethod
    def _collect_from_directory(cls, directory: Path) -> list[Path]:
        """Primitive method for collect_prototypes.

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

    def collect_prototypes(self) -> list[Path]:
        """Collects modules from the source prototypes directory."""
        if not self.prototypes_source_path.exists():
            relative_path = self.prototypes_source_path.relative_to(self.path)
            msg = f"Project {self.name} does not contain the requisite {relative_path} directory."
            raise FileNotFoundError(msg)
        return self._collect_from_directory(self.prototypes_source_path)

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

    def import_prototypes(self) -> list[ModuleType]:
        """Imports the modules in the project's src/<project>/api/prototypes_ directory."""
        if not self.api_prototypes_path.exists():
            relative_path = self.api_prototypes_path.relative_to(self.path)
            msg = f"Project {self.name} does not contain the requisite {relative_path} directory."
            raise FileNotFoundError(msg)
        # TODO: this should be unnecessary -only self.path/src needs to be added
        if self.api_prototypes_path not in sys.path:
            sys.path.insert(0, self.api_prototypes_path.as_posix())
        modules = self._import_from_directory(self.api_prototypes_path)
        # Next we validate that no import targets the original modules in the source prototypes
        # directory, which could be the case if absolute imports are used
        for name, module in sys.modules.items():
            try:
                module_path = Path(module.__file__)  # type: ignore
            except (AttributeError, TypeError, ValueError):
                continue
            if module_path.is_relative_to(self.prototypes_source_path):
                msg = (
                    "Modules in the prototypes directory that import other modules in that "
                    f"directory must use relative syntax. {name} in module "
                    f"{module.__name__} does not."
                )
                raise ImportError(msg)
        return modules

    def copy_prototypes(self):
        """Copies modules from the src/prototypes directory to the application directory."""
        models_path = self.prototypes_source_path
        copy_path = self.api_prototypes_path
        for module_path in self.collect_prototypes():
            relative_path = module_path.relative_to(models_path)
            destination = copy_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            module_path.copy(destination)

    def compile(self, *compilations: str, **kwargs):
        """Compiles the project using the specified compilers."""
        return None
