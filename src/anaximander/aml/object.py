# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Define the base AML object archetype."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from .archetype import archetype
from .prototype import Arche, prototype

# endregion

# =============================================================================
# Object archetype
# =============================================================================
# region Object archetype


@archetype
class Object(Arche, metaclass=prototype):
    """Represent the base AML archetype for objects."""
    pass

# endregion
