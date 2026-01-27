"""Archetypes for AML models."""

# =============================================================================
# Imports
# =============================================================================

from typing import dataclass_transform

from .archetype import archetype
from .metadescriptors import Metadescriptor
from .object import Object
from .protodescriptors import (
    AssignableFieldProtodescriptor,
    ConstructorDeclarator,
    FieldProtodescriptor,
    SchemaDeclarator,
)

# =============================================================================
# Model archetype
# =============================================================================


@archetype
@dataclass_transform(field_specifiers=(AssignableFieldProtodescriptor,))
class Model(Object):
    """Base archetype for AML models with fields and schema descriptors."""

    __declarator_types__ = {
        Metadescriptor,
        ConstructorDeclarator,
        FieldProtodescriptor,
        SchemaDeclarator,
    }
