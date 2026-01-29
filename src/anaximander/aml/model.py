"""Archetypes for AML models."""

# =============================================================================
# Imports
# =============================================================================

from typing import dataclass_transform

from .archetype import archetype
from .declarators import (
    AssignableFieldProtodescriptor,
    ConstructorDeclarator,
    FieldProtodescriptor,
    Metadescriptor,
    SchemaDeclarator,
)
from .object import Object

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
