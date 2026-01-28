import sys
from pathlib import Path

import anaximander.aml as nx


def test_finalize_module_resolves_forward_refs(tmp_path):
    module_path = tmp_path / "hints_module.py"
    module_path.write_text(
        """
from __future__ import annotations

import anaximander.aml as nx

class Temperature(nx.Measurement):
    pass

class Sensor(nx.Model):
    temperature: Temperature | None = nx.data()
"""
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = __import__("hints_module")
        nx.finalize_module(module)
        sensor = module.Sensor
        field = sensor.metacharacters("merged").field["temperature"]
        assert field.type is module.Temperature
        assert field.nullable is True
    finally:
        sys.path.remove(str(tmp_path))
