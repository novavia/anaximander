# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Expose the top-level Anaximander public API."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from .aml import Object, archetype, trait

# endregion

# =============================================================================
# Public exports
# =============================================================================
# region Public exports

__all__ = [
    "archetype",
    "trait",
    "Object",
]

# endregion
