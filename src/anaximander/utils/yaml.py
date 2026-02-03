# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Provide YAML serialization helpers with NX-specific defaults."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import re

import yaml

# endregion

# =============================================================================
# Dumper/loader state
# =============================================================================
# region Dumper/loader state

NX_YAML_DUMPER: type[yaml.SafeDumper] | None = None
NX_YAML_LOADER: type[yaml.SafeLoader] | None = None
NX_YAML_TYPES: dict[str, type] = {}

# endregion

# =============================================================================
# Dumper factory
# =============================================================================
# region Dumper factory


def _build_dumper() -> type[yaml.SafeDumper]:
    """Build a YAML dumper with NX-wide defaults.

    Returns:
        The configured SafeDumper subclass.
    """

    class NxYamlDumper(yaml.SafeDumper):
        """NX-wide YAML dumper with readable formatting."""

        def increase_indent(self, flow=False, indentless=False):
            """Increase indentation while always allowing nested flow content."""
            return super().increase_indent(flow, indentless=False)

    # Literals that YAML would coerce unless quoted.
    _ambiguous_literals = {
        "~",
        "null",
        "Null",
        "NULL",
        "true",
        "True",
        "TRUE",
        "false",
        "False",
        "FALSE",
    }
    # Numeric-looking strings are quoted to prevent coercion.
    _numeric_pattern = re.compile(
        r"^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$|^[+-]?\.\d+(?:[eE][+-]?\d+)?$"
    )

    def _repr_str(dumper: yaml.SafeDumper, obj: str):
        """Represent strings while guarding against ambiguous literals."""
        # Quote ambiguous string scalars to preserve round-trip semantics.
        if obj in _ambiguous_literals or _numeric_pattern.fullmatch(obj):
            return dumper.represent_scalar("tag:yaml.org,2002:str", obj, style='"')
        return dumper.represent_scalar("tag:yaml.org,2002:str", obj)

    NxYamlDumper.add_representer(str, _repr_str)
    return NxYamlDumper


def _build_loader() -> type[yaml.SafeLoader]:
    """Build a YAML loader with NX-wide defaults.

    Returns:
        The configured SafeLoader subclass.
    """

    class NxYamlLoader(yaml.SafeLoader):
        """NX-wide YAML loader."""
        pass

    return NxYamlLoader


def _ensure_dumper() -> type[yaml.SafeDumper]:
    """Return the NX-wide YAML dumper class, creating it if needed."""
    global NX_YAML_DUMPER
    if NX_YAML_DUMPER is None:
        NX_YAML_DUMPER = _build_dumper()
    return NX_YAML_DUMPER


def _ensure_loader() -> type[yaml.SafeLoader]:
    """Return the NX-wide YAML loader class, creating it if needed."""
    global NX_YAML_LOADER
    if NX_YAML_LOADER is None:
        NX_YAML_LOADER = _build_loader()
    return NX_YAML_LOADER


def nx_register_representer(target: type, representer) -> None:
    """Register a representer on the NX-wide YAML dumper.

    Args:
        target: Target Python type to represent.
        representer: Callable that builds YAML nodes for the type.
    """
    _ensure_dumper().add_representer(target, representer)


def nx_register_multi_representer(target: type, representer) -> None:
    """Register a multi-representer on the NX-wide YAML dumper.

    Args:
        target: Target Python type to represent.
        representer: Callable that builds YAML nodes for the type.
    """
    _ensure_dumper().add_multi_representer(target, representer)


def nx_register_constructor(tag: str, constructor) -> None:
    """Register a constructor on the NX-wide YAML loader.

    Args:
        tag: YAML tag to register.
        constructor: Callable that constructs values for the tag.
    """
    _ensure_loader().add_constructor(tag, constructor)


def nx_register_multi_constructor(tag: str, constructor) -> None:
    """Register a multi-constructor on the NX-wide YAML loader.

    Args:
        tag: YAML tag prefix to register.
        constructor: Callable that constructs values for the tag.
    """
    _ensure_loader().add_multi_constructor(tag, constructor)


def nx_register_type(name: str, type_: type) -> None:
    """Register a custom type name for YAML resolution.

    Args:
        name: Type name used in YAML.
        type_: Python type to resolve.
    """
    NX_YAML_TYPES[name] = type_


def nx_set_ignore_aliases(predicate) -> None:
    """Extend the dumper's ignore_aliases logic with a custom predicate.

    Args:
        predicate: Callable that returns True when aliases should be ignored.
    """
    dumper = _ensure_dumper()
    base = dumper.ignore_aliases

    def _ignore_aliases(self, data):
        """Return True when aliases should be suppressed."""
        if predicate(data):
            return True
        return base(self, data)

    dumper.ignore_aliases = _ignore_aliases


def nx_yaml_dump(data: object) -> str:
    """Serialize to YAML with NX-wide defaults.

    Args:
        data: Data to serialize.

    Returns:
        YAML string representation.
    """
    dumper = _ensure_dumper()
    return yaml.dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        Dumper=dumper,
        indent=2,
    )


def nx_yaml_load(data: str) -> object:
    """Load YAML with NX-wide defaults.

    Args:
        data: YAML string to parse.

    Returns:
        Parsed Python object.
    """
    loader = _ensure_loader()
    return yaml.load(data, Loader=loader)

# endregion
