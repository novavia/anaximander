import sys
from pathlib import Path

import anaximander.aml as nx


def test_smoke_finalize_module(tmp_path):
    module_path = tmp_path / "smoke_module.py"
    module_path.write_text(
        """
import anaximander.aml as nx

class Temperature(nx.Measurement):
    pass

class Sensor(nx.Model):
    temperature: Temperature = nx.data()
"""
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = __import__("smoke_module")
        nx.finalize_module(module)
        assert any(cls.__name__ == "Temperature" for cls in module.__prototypes__)
        assert any(cls.__name__ == "Sensor" for cls in module.__prototypes__)
    finally:
        sys.path.remove(str(tmp_path))
