# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Exercise KwargMap behavior and Config integration with OmegaConf."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from dataclasses import dataclass

import omegaconf
import pytest

from anaximander.utils import Config, KwargMap

# endregion

# =============================================================================
# Tests
# =============================================================================
# region Tests


def test_kwarg_map():
    """Validate KwargMap contextual lookup, ordering, sorting, and subselection."""
    km_0 = KwargMap(a=1, b=2)
    assert dict(km_0) == {"a": 1, "b": 2}
    assert str(km_0) == "a=1, b=2"

    # Only string keys are accepted.
    with pytest.raises(TypeError):
        km_0[1] = 2  # type: ignore

    # Use a mapping as context.
    km_1 = KwargMap(km_0, a=".a")
    assert km_1["a"] == 1
    assert str(km_1) == "a=1"

    # Use implicit context (locals).
    km_2 = KwargMap(a=".km_0")
    assert km_2["a"] == km_0

    # Use an object as context.
    @dataclass
    class C:
        """Simple context class for attribute resolution tests."""

        x: int
        y: int

    # Test the "_" key.
    km_3 = KwargMap(C(1, 2), _=["x", "y"])
    assert km_3["x"] == 1
    assert km_3["y"] == 2

    # Test key deletion.
    del km_3["x"]
    assert str(km_3) == "y=2"
    with pytest.raises(KeyError):
        del km_3["x"]

    # Test insertion, including redundant keys.
    km_3["y"] = ".y"
    assert km_3.data == {"y": ".y"}
    km_3["_"] = ["x"]
    assert str(km_3) == "y=2, x=1"

    # Test sorting.
    km_3.sort()
    assert str(km_3) == "x=1, y=2"
    km_3.sort(["y", "x"])
    assert str(km_3) == "y=2, x=1"
    km_3.sort(["y", "x"], reverse=True)
    assert str(km_3) == "x=1, y=2"
    with pytest.raises(KeyError):
        km_3.sort(["x", "y", "z"])

    # Test calling.
    km_4 = km_3("x")
    assert str(km_4) == "x=1"


@pytest.mark.bugger
def test_config():
    """Ensure Config subclasses integrate with OmegaConf via the omegaconf property."""

    class MyConfig(Config):
        """Sample Config subclass for OmegaConf integration."""

        x: int
        y: int = 0

    mc = MyConfig(x=1)
    conf = mc.omegaconf
    assert isinstance(conf, omegaconf.DictConfig)

# endregion
