# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Provide helpers for creating and managing Anaximander projects."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import ast
import shutil
import sys
from importlib import import_module
import pkgutil
from pathlib import Path
from typing import cast

import attrs
from cookiecutter.main import cookiecutter

from .aml.diagnostics import DiagnosticBag, PROJECT_DIAGNOSTICS, raise_on_errors
from .aml.modules import NxModuleType, finalize_module
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
# Project configuration
# =============================================================================
# region Project configuration


class ProjectConfig(Config):
    """Declare configuration for project scaffolding."""
    name: str

    @property
    def slug(self) -> str:
        """Return the normalized slug for the project name."""
        return self.name.lower().replace(" ", "_").replace("-", "_")

    def cookiecutter_context(self):
        """Return the Cookiecutter context for project scaffolding."""
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
    """Represent an Anaximander project and its on-disk layout."""

    path: Path = attrs.field(converter=Path)
    _config: ProjectConfig | None = private_field(default=None)
    __diagnostics__: DiagnosticBag = private_field()

    def __attrs_post_init__(self):
        """Finalize initialization by registering the import path."""
        self.__diagnostics__ = DiagnosticBag(self)
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
        """Create a new Anaximander project in the specified directory.

        Args:
            project_name: A name for the project, which also sets the name of the project
                directory. It can be capitalized, but note that the name is converted to lowercase
                to define a top-level Python package.
            parent_directory: The directory in which to create
                the project. It can be absolute or relative to the current working directory.
                If None (the default), then it is set to the current working directory.
            domain: An optional path to either a python file
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
            A new Anaximander project instance.

        Raises:
            FileNotFoundError: If the domain path does not exist.
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
        # Instantiate and persist project configuration.
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
        """Return the project directory name."""
        return self.path.name

    @property
    def slug(self):
        """Return the configured project slug."""
        return self.config.slug

    @property
    def config_path(self) -> Path:
        """Return the path to the project configuration file."""
        return self.path / ".nxp"

    @property
    def config(self) -> ProjectConfig:
        """Return the cached project configuration, loading if needed."""
        if not self._config:
            self._config = ProjectConfig.load(self.config_path)
        return self._config

    @property
    def code_path(self) -> Path:
        """Return the project code root path."""
        return self.path / "src"

    @property
    def domain_source_path(self) -> Path:
        """Return the domain package root path."""
        return self.path / "src/domain"

    @property
    def application_path(self) -> Path:
        """Return the application package root path."""
        return self.path / f"src/{self.slug}"

    @property
    def compilation_path(self) -> Path:
        """Return the compilation output path."""
        return self.application_path / "api"

    def set_import_path(self):
        """Set the project's code path on the interpreter's import path."""
        if (code_path := self.code_path).exists():
            if code_path not in sys.path:
                sys.path.insert(0, code_path.as_posix())

    @classmethod
    def is_project_directory(cls, path: Path | str) -> bool:
        """Return whether the given path is a project directory."""
        return (Path(path) / ".nxp").exists()

    @classmethod
    def _collect_from_directory(cls, directory: Path) -> list[Path]:
        """Collect Python module paths from a directory tree.

        Args:
            directory: The directory containing python files to import.

        Returns:
            A list of module file paths.
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
        """Collect modules from the source domain directory."""
        if not self.domain_source_path.exists():
            relative_path = self.domain_source_path.relative_to(self.path)
            msg = f"Project {self.name} does not contain the requisite {relative_path} directory."
            raise FileNotFoundError(msg)
        return self._collect_from_directory(self.domain_source_path)

    @classmethod
    def _import_from_directory(
        cls, directory: Path, package: str | None = None
    ) -> list[NxModuleType]:
        """Import modules recursively from a directory tree.

        Args:
            directory: The directory containing python files to import.
            package: Optional parent package, possibly nested.
                Defaults to None.

        Returns:
            A list of imported modules.
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
            # Preserve module AST for AML analysis.
            module.__ast__ = module_ast
            modules.append(module)
        return modules

    def import_domain(self) -> list[NxModuleType]:
        """Import the modules in the project's src/domain directory."""
        if not self.domain_source_path.exists():
            relative_path = self.domain_source_path.relative_to(self.path)
            msg = f"Project {self.name} does not contain the requisite {relative_path} directory."
            raise FileNotFoundError(msg)
        package = "domain"
        modules = self._import_from_directory(self.domain_source_path, package=package)
        return modules

    def load_domain(self) -> list[NxModuleType]:
        """Import and finalize all modules in the project's domain package."""
        if not self.domain_source_path.exists():
            relative_path = self.domain_source_path.relative_to(self.path)
            msg = f"Project {self.name} does not contain the requisite {relative_path} directory."
            raise FileNotFoundError(msg)
        token = PROJECT_DIAGNOSTICS.set(self.__diagnostics__)
        try:
            self.set_import_path()
            domain_pkg = import_module("domain")
            modules: list[NxModuleType] = [cast(NxModuleType, domain_pkg)]
            for module_info in pkgutil.walk_packages(domain_pkg.__path__, prefix="domain."):
                module = cast(NxModuleType, import_module(module_info.name))
                modules.append(module)
            for module in modules:
                finalize_module(module)
            raise_on_errors(self.__diagnostics__)
            return modules
        finally:
            PROJECT_DIAGNOSTICS.reset(token)

    def compile(self, *compilations: str, **kwargs):
        """Compile the project using the specified compilers."""
        return None

# endregion
