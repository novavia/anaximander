from __future__ import annotations

from typing import Any, cast

from anaximander.aml.declarators import MISSING, Declarator, register_yaml_type
from anaximander.utils.yaml import nx_yaml_dump, nx_yaml_load
from tests.anaximander.aml.modules import value_parsers_module as vpm


def _pick_declarator() -> Declarator:
    registries = vpm.Sensor.__merged_declarators__
    for handle in ("field", "metadata", "option", "nxfield", "schema", "constructor"):
        registry = getattr(registries, handle, None)
        if registry:
            for item in registry.values():
                return item
    raise AssertionError("No declarator found for serdes tests.")


def test_aml_yaml_missing_roundtrip():
    payload = {"missing": MISSING}
    dumped = nx_yaml_dump(payload)
    assert "!Missing" in dumped
    loaded = cast(dict[str, Any], nx_yaml_load(dumped))
    assert loaded["missing"] is MISSING


def test_aml_yaml_prototype_roundtrip(aml_finalize):
    aml_finalize(vpm)
    payload = {"proto": vpm.Sensor}
    dumped = nx_yaml_dump(payload)
    try:
        loaded = cast(dict[str, Any], nx_yaml_load(dumped))
    except KeyError as exc:
        assert "Ambiguous type 'Sensor'" in str(exc)
        qualified = f"<{vpm.Sensor.__project__}::{vpm.Sensor.__module__}::{vpm.Sensor.__name__} prototype>"  # noqa
        loaded = cast(dict[str, Any], nx_yaml_load(f"proto: {qualified}\n"))
    assert loaded["proto"] is vpm.Sensor


def test_aml_yaml_declarator_roundtrip(aml_finalize):
    aml_finalize(vpm)
    declarator = _pick_declarator()
    payload = {"decl": declarator}
    dumped = nx_yaml_dump(payload)
    loaded = cast(dict[str, Any], nx_yaml_load(dumped))
    assert loaded["decl"] is declarator


def test_aml_yaml_type_roundtrip():
    payload = {"type": int}
    dumped = nx_yaml_dump(payload)
    loaded = cast(dict[str, Any], nx_yaml_load(dumped))
    assert loaded["type"] is int


def test_aml_yaml_custom_type_roundtrip():
    class Custom:
        pass

    register_yaml_type("Custom", Custom)
    payload: dict[str, Any] = {"custom": "<Custom>"}
    dumped = nx_yaml_dump(payload)
    loaded = cast(dict[str, Any], nx_yaml_load(dumped))
    assert loaded["custom"] is Custom
