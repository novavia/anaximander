"""Archetypes for AML models."""

# =============================================================================
# Imports
# =============================================================================

from anaximander.aml.archetype import archetype
from anaximander.aml.metadescriptors import Metadescriptor
from anaximander.aml.object import Object
from anaximander.aml.protodescriptors import (
    ConstructorDeclarator,
    FieldProtodescriptor,
    SchemaDeclarator,
)
from anaximander.aml.prototype import prototype

# =============================================================================
# Model archetype
# =============================================================================


@archetype
class Model(Object, metaclass=prototype):
    """Base archetype for AML models with fields and schema descriptors."""

    __declarator_types__ = {
        Metadescriptor,
        ConstructorDeclarator,
        FieldProtodescriptor,
        SchemaDeclarator,
    }
