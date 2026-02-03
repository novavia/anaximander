# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Define archetypes and helpers for AML models."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from typing import TypeAliasType, dataclass_transform, get_args, get_origin

from ..utils.funcs import unwrap_optional_type
from .archetype import archetype
from .basetypes import is_pydata_type
from .data import Data
from .declarators import (
    AssignableFieldProtodescriptor,
    ConstructorDeclarator,
    DataProtodescriptor,
    FieldProtodescriptor,
    Metadescriptor,
    NxFieldDeclarator,
    SchemaDeclarator,
)
from .object import Object
from .prototype import prototype

# endregion

# =============================================================================
# Model archetype
# =============================================================================
# region Model archetype


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

    @classmethod
    def _validate_nxfields(cls, owner: prototype) -> None:
        """Validate nxfield bindings for a concrete model owner.

        Args:
            owner: Prototype instance to validate.

        Raises:
            TypeError: If bindings are invalid or reference incompatible fields.
            KeyError: If bindings reference missing fields.
        """
        bindings = owner.__merged_bindings__
        declarators = owner.__merged_declarators__
        if not bindings.nxfield:
            return
        field_registry = declarators.field
        for name, target in bindings.nxfield.items():
            declarator = bindings.nxfield.declarators[name]
            if not isinstance(declarator, NxFieldDeclarator):
                continue
            if not isinstance(target, str):
                raise TypeError(f"Nxfield '{name}' in '{owner.__name__}' must bind to a str.")
            if target not in field_registry:
                raise KeyError(
                    f"Nxfield '{name}' in '{owner.__name__}' references unknown field '{target}'."
                )
            field = field_registry[target]
            hint = field.hint if field.hint is not None else field.type
            if hint is None:
                continue
            hint, _ = unwrap_optional_type(hint)
            if isinstance(hint, type) and not issubclass(hint, declarator.fieldtype):
                raise TypeError(
                    f"Nxfield '{name}' in '{owner.__name__}' expects "
                    f"{declarator.fieldtype.__name__} field, got {hint.__name__}."
                )

    @classmethod
    def _validate_data_fields(cls, owner: prototype) -> None:
        """Validate data field annotations for a concrete model owner.

        Args:
            owner: Prototype instance to validate.

        Raises:
            TypeError: If data field annotations violate AML expectations.
        """
        declarators = owner.__merged_declarators__
        field_registry = declarators.field

        def _unwrap_alias(tp: object) -> object:
            """Unwrap TypeAliasType values for consistent checks."""
            return tp.__value__ if isinstance(tp, TypeAliasType) else tp

        def _is_datatype_hint(tp: object) -> bool:
            """Whether a hint resolves to a Data archetype or PyData."""
            tp = _unwrap_alias(tp)
            if isinstance(tp, type) and issubclass(tp, Data):
                return True
            return is_pydata_type(tp)

        def _is_modeltype_hint(tp: object) -> bool:
            """Whether a hint is a strict Model archetype (not Entity/Record/Document)."""
            tp = _unwrap_alias(tp)
            if not isinstance(tp, type) or not issubclass(tp, Model):
                return False
            return getattr(tp, "__archetype__", None) is Model

        def _is_collection_datatype_hint(tp: object) -> bool:
            """Whether a hint is a built-in collection of DataType elements."""
            tp = _unwrap_alias(tp)
            origin = get_origin(tp)
            if origin is list or origin is set:
                args = get_args(tp)
                # Homogeneous list/set of DataType
                return len(args) == 1 and _is_datatype_hint(args[0])
            if origin is tuple:
                args = get_args(tp)
                if len(args) == 2 and args[1] is Ellipsis:
                    # Homogeneous tuple of DataType
                    return _is_datatype_hint(args[0])
                if len(args) == 1:
                    # Accept 1-arg tuple aliases as homogeneous
                    return _is_datatype_hint(args[0])
                return False
            if origin is dict:
                args = get_args(tp)
                # Dict keys must be int or str; values must be DataType
                if len(args) != 2 or args[0] not in (int, str):
                    return False
                return _is_datatype_hint(args[1])
            return False

        for field in field_registry.values():
            if not isinstance(field, DataProtodescriptor):
                continue
            hint = field.hint if field.hint is not None else field.type
            # Data descriptors must always declare a type hint.
            if hint is None:
                raise TypeError(
                    f"Data field '{field.name}' in '{owner.__name__}' "
                    "must specify a type annotation."
                )
            hint, _ = unwrap_optional_type(hint)
            # Admissible data types:
            # - Data archetypes (DataType)
            # - PyData (plain Python types)
            # - Strict Model archetypes (embedded submodels)
            # - Built-in homogeneous collections of DataType
            if _is_datatype_hint(hint) or _is_modeltype_hint(hint) or _is_collection_datatype_hint(hint):  # noqa
                continue
            raise TypeError(
                f"Invalid data field annotation for '{field.name}' in '{owner.__name__}'."
            )

    @classmethod
    def __validate_declarators__(cls, owner: type) -> None:
        """Validate model-specific declarators for an owner."""
        # Owner-aware validation for declarators that reference model fields or types.
        cls._validate_nxfields(owner)
        cls._validate_data_fields(owner)

# endregion
