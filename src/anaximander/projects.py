"""Utilities for creating and managing Anaximander projects."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import ast
import shutil
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import cast

import attrs
from cookiecutter.main import cookiecutter

from .aml import prototype
from .utils import Config, private_field
from .utils.funcs import workdir

# endregion

# =============================================================================
# Constants
# =============================================================================
# region Constants

NXPATH = Path(__file__).parent
PROJECT_TEMPLATE = NXPATH / "config/projects/project_template"

# endregion

# =============================================================================
# Module types
# =============================================================================
# region Module types


class NxModuleType(ModuleType):
    """A type hint for Anaximander AML declarative modules."""
    __ast__: ast.Module  # Holds the module's parsed abstract syntax tree
    __prototypes__: list[prototype]  # Holds the module's declared types

# endregion

# =============================================================================
# Project configuration
# =============================================================================
# region Project configuration


class ProjectConfig(Config):
    name: str

    @property
    def slug(self) -> str:
        return self.name.lower().replace(' ', '_').replace('-', '_')

    def cookiecutter_context(self):
        return {
            "project_name": self.name,
            "project_slug": self.slug,
        }

# endregion

# =============================================================================
# Project model
# =============================================================================
# region Project model


@attrs.define
class Project:
    """A class that represents an Anaximander project."""

    path: Path = attrs.field(converter=Path)
    _config: ProjectConfig | None = private_field(default=None)

    def __attrs_post_init__(self):
        self.set_import_path()

    @classmethod
    def create(
        cls,
        project_name: str,
        parent_directory: str | Path | None = None,
        *,
        domain: str | Path | None = None,
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
            domain (str | Path | None, optional): An optional path to either a python file
                or directory containing domain model declarations. Like parent_directory it can be
                either absolute or relative to the current working directory.
                Note that the name of the file or directory is ignored:
                - If the path points to a python file, it is turned to 'src/domain/__init__.py'
                - If the path points to a directory, its content is copied to 'src/domain'
                Hence if the domain models are defined in a module my_domain.py and the name
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
        if domain is not None:
            domain = Path(domain)
            if not domain.exists():
                raise FileNotFoundError(f"Domain path {domain} does not exist.")
            if domain.is_dir():
                destination = project.domain_source_path
                shutil.copytree(domain, destination, dirs_exist_ok=True)
            else:
                destination = project.domain_source_path / "__init__.py"
                shutil.copy(domain, destination)
        return project

    @property
    def name(self):
        return self.path.name

    @property
    def slug(self):
        return self.config.slug

    @property
    def config_path(self) -> Path:
        return self.path / ".nxp"

    @property
    def config(self) -> ProjectConfig:
        if not self._config:
            self._config = ProjectConfig.load(self.config_path)
        return self._config

    @property
    def code_path(self) -> Path:
        return self.path / "src"

    @property
    def domain_source_path(self) -> Path:
        return self.path / "src/domain"

    @property
    def application_path(self) -> Path:
        return self.path / f"src/{self.slug}"

    @property
    def compilation_path(self) -> Path:
        return self.application_path / "api"

    def set_import_path(self):
        """Sets the project's code path on the interpreter's import path."""
        if (code_path := self.code_path).exists():
            if code_path not in sys.path:
                sys.path.insert(0, code_path.as_posix())

    @classmethod
    def is_project_directory(cls, path: Path | str) -> bool:
        return (Path(path) / ".nxp").exists()

    @classmethod
    def _collect_from_directory(cls, directory: Path) -> list[Path]:
        """Primitive method for collect_domain.

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

    def collect_domain(self) -> list[Path]:
        """Collects modules from the source domain directory."""
        if not self.domain_source_path.exists():
            relative_path = self.domain_source_path.relative_to(self.path)
            msg = f"Project {self.name} does not contain the requisite {relative_path} directory."
            raise FileNotFoundError(msg)
        return self._collect_from_directory(self.domain_source_path)

    @classmethod
    def _import_from_directory(
        cls, directory: Path, package: str | None = None
    ) -> list[NxModuleType]:
        """Primitive function for import_domain.

        Args:
            directory (Path): The directory containing python files to import.
            package (str | None, optional): Optional parent package, possibly nested.
                Defaults to None.

        Returns:
            list[NxModuleType]: A list of imported modules.
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
            module_code = module_path.read_text()
            module_ast = ast.parse(module_code)
            module_name = module_path.stem
            if package is not None:
                module_name = "." + module_name
            module: NxModuleType = cast(NxModuleType, import_module(module_name, package=package))
            module.__ast__ = module_ast
            modules.append(module)
        return modules

    def import_domain(self) -> list[NxModuleType]:
        """Imports the modules in the project's src/domain directory."""
        if not self.domain_source_path.exists():
            relative_path = self.domain_source_path.relative_to(self.path)
            msg = f"Project {self.name} does not contain the requisite {relative_path} directory."
            raise FileNotFoundError(msg)
        package = "domain"
        modules = self._import_from_directory(self.domain_source_path, package=package)
        return modules

    def compile(self, *compilations: str, **kwargs):
        """Compiles the project using the specified compilers."""
        return None

# endregion
