"""YAML serialization helpers."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import re

import yaml

# endregion

# =============================================================================
# Dumper factory
# =============================================================================
# region Dumper factory


NX_YAML_DUMPER = None
NX_YAML_LOADER = None
NX_YAML_TYPES: dict[str, type] = {}


def _build_dumper() -> type[yaml.SafeDumper]:
    """Build a YAML dumper with NX-wide defaults."""

    class NxYamlDumper(yaml.SafeDumper):
        """NX-wide YAML dumper with readable formatting."""

        def increase_indent(self, flow=False, indentless=False):
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
        "y",
        "Y",
        "yes",
        "Yes",
        "YES",
        "n",
        "N",
        "no",
        "No",
        "NO",
        "on",
        "On",
        "ON",
        "off",
        "Off",
        "OFF",
    }
    # Numeric-looking strings are quoted to prevent coercion.
    _numeric_pattern = re.compile(
        r"^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$|^[+-]?\.\d+(?:[eE][+-]?\d+)?$"
    )

    def _repr_str(dumper: yaml.SafeDumper, obj: str):
        # Quote ambiguous string scalars to preserve round-trip semantics.
        if obj in _ambiguous_literals or _numeric_pattern.fullmatch(obj):
            return dumper.represent_scalar("tag:yaml.org,2002:str", obj, style='"')
        return dumper.represent_scalar("tag:yaml.org,2002:str", obj)

    NxYamlDumper.add_representer(str, _repr_str)
    return NxYamlDumper


def _build_loader() -> type[yaml.SafeLoader]:
    """Build a YAML loader with NX-wide defaults."""

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
    """Register a representer on the NX-wide YAML dumper."""
    _ensure_dumper().add_representer(target, representer)


def nx_register_multi_representer(target: type, representer) -> None:
    """Register a multi-representer on the NX-wide YAML dumper."""
    _ensure_dumper().add_multi_representer(target, representer)


def nx_register_constructor(tag: str, constructor) -> None:
    """Register a constructor on the NX-wide YAML loader."""
    _ensure_loader().add_constructor(tag, constructor)


def nx_register_multi_constructor(tag: str, constructor) -> None:
    """Register a multi-constructor on the NX-wide YAML loader."""
    _ensure_loader().add_multi_constructor(tag, constructor)


def nx_register_type(name: str, type_: type) -> None:
    """Register a custom type name for YAML resolution."""
    NX_YAML_TYPES[name] = type_


def nx_yaml_dump(data: object) -> str:
    """Serialize to YAML with NX-wide defaults."""
    dumper = _ensure_dumper()
    return yaml.dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        Dumper=dumper,
        indent=2,
    )


def nx_yaml_load(data: str) -> object:
    """Load YAML with NX-wide defaults."""
    loader = _ensure_loader()
    return yaml.load(data, Loader=loader)

# endregion
