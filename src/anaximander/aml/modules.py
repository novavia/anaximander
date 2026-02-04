# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Finalize AML modules after import.

Module finalization is the bridge between declarative syntax and executable
prototype semantics. This module parses module source, enforces a strict AST
allow list, binds AST nodes to declarators and prototypes, resolves forward
references, and validates core archetype/trait/prototype invariants.

It also registers prototypes into a module registry used by YAML loaders and
compiler pipelines, ensuring that declarative models can be referenced by
qualified names across project boundaries.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import ast
import builtins
import re
from pathlib import Path
from types import ModuleType
from typing import cast, get_type_hints

import yaml

from ..utils.funcs import unwrap_classvar_type, unwrap_optional_type
from ..utils.yaml import NX_YAML_TYPES, nx_register_constructor
from .data import Data
from .declarators import (
    MISSING,
    AnnotatableDeclarator,
    Declarator,
    EnumerationCallableDeclarator,
    FieldGroupProtodescriptor,
    ParserDeclarator,
    PrototypeValidator,
    ValidatorDeclarator,
    is_not_missing,
)
from .diagnostics import (
    BindingInvalidValue,
    DECLARATOR_DIAGNOSTICS,
    DisallowedModuleStatement,
    EnumerationInvalid,
    InvalidTypeRole,
    MODULE_DIAGNOSTICS,
    ModuleDiagnosticBag,
    ModuleSourceMissing,
    PROTOTYPE_DIAGNOSTICS,
    PrototypeMetadescriptorForbidden,
    PrototypeValidatorFailed,
    ReferenceInvalid,
    ReferenceKindInvalid,
    ReferenceMissingOwnerMember,
    TypeHintResolutionFailed,
    ValidatorFailed,
    diagnostic_context,
    raise_on_errors,
)
from .model import Model
from .prototype import TypeRole, is_archetype, is_prototype, is_trait, prototype
from .registries import DeclarativeTypeRegistry

# endregion

# =============================================================================
# Module types
# =============================================================================
# region Module types


class NxModuleType(ModuleType):
    """A type hint for Anaximander AML declarative modules."""

    __ast__: ast.Module  # Holds the module's parsed abstract syntax tree
    __diagnostics__: ModuleDiagnosticBag  # Holds module-scoped diagnostics
    __prototypes__: list[prototype]  # Holds the module's declared types
    __finalized__: bool  # Whether the module has been finalized

# endregion

# =============================================================================
# Constants
# =============================================================================
# region Constants

DEFAULT_ALLOWED_MODULE_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.Expr,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.If,
    ast.Pass,
)

DEFAULT_ALLOWED_CLASS_NODES: tuple[type[ast.AST], ...] = (
    ast.Assign,
    ast.AnnAssign,
    ast.Expr,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Pass,
)

# endregion

# =============================================================================
# Prototype registry
# =============================================================================
# region Prototype registry


class PrototypeRegistry(DeclarativeTypeRegistry[NxModuleType, prototype]):
    """Registry for prototypes and their owning modules."""

    def _resolve_from_module(self, module: NxModuleType, name: str) -> prototype:
        if hasattr(module, "__prototypes__"):
            for proto in getattr(module, "__prototypes__", []):
                if proto.__name__ == name:
                    return proto
        return module.__dict__[name]


PROTOTYPES = PrototypeRegistry()

# endregion

# =============================================================================
# YAML loader hooks
# =============================================================================
# region YAML loader hooks


_REF_PATTERN = re.compile(r"^<(?P<body>.+)>$")
_QUALIFIED_PATTERN = re.compile(r"^(?P<project>[^:]+)::(?P<module>[^:]+)::(?P<name>.+)$")
_KIND_SUFFIXES = (
    " data field",
    " prototype",
    " callable",
    " field",
    " metadata",
    " option",
    " nxfield",
    " schema",
    " constructor",
    " type",
)


def _parse_reference(value: str) -> tuple[str | None, str | None, str]:
    match = _QUALIFIED_PATTERN.match(value)
    if match:
        return match.group("project"), match.group("module"), match.group("name")
    return None, None, value


def _split_owner(ref: str) -> tuple[str, str | None]:
    if "." in ref:
        owner_ref, member = ref.split(".", 1)
        return owner_ref, member
    return ref, None


def _split_kind(body: str) -> tuple[str, str | None]:
    for suffix in _KIND_SUFFIXES:
        if body.endswith(suffix):
            return body[:-len(suffix)], suffix.strip()
    return body, None


def _resolve_prototype(name: str, *, project: str | None = None, module: str | None = None) -> prototype:  # noqa
    return PROTOTYPES.resolve_type(name, project=project, module=module)


def _resolve_tagged_reference(ref: str, kind: str) -> object | None:
    if kind == "prototype":
        project, module, name = _parse_reference(ref)
        return _resolve_prototype(name, project=project, module=module)
    if kind in {"data field", "field", "metadata", "option", "nxfield", "schema", "constructor"}:
        owner_ref, member = _split_owner(ref)
        if member is None:
            ReferenceMissingOwnerMember.report(
                f"Declarator reference '{ref} {kind}' must include owner and member."
            )
            return None
        project, module, owner_name = _parse_reference(owner_ref)
        owner = _resolve_prototype(owner_name, project=project, module=module)
        return getattr(owner, member)
    if kind == "callable":
        owner_ref, member = _split_owner(ref)
        if member is None:
            ReferenceMissingOwnerMember.report(
                f"Callable reference '{ref} {kind}' must include owner and member."
            )
            return None
        project, module, owner_name = _parse_reference(owner_ref)
        owner = _resolve_prototype(owner_name, project=project, module=module)
        return getattr(owner, member)
    if kind == "type":
        project, module, name = _parse_reference(ref)
        if name in builtins.__dict__:
            return builtins.__dict__[name]
        if name in NX_YAML_TYPES:
            return NX_YAML_TYPES[name]
        if project is None and module is None:
            ReferenceKindInvalid.report(f"Unknown type reference '{ref} {kind}'.")
            return None
        return _resolve_prototype(name, project=project, module=module)
    ReferenceKindInvalid.report(f"Unknown reference kind '{kind}'.")
    return None


def _resolve_reference(value: str) -> object | None:
    match = _REF_PATTERN.match(value)
    if not match:
        if value == "MISSING":
            return MISSING
        return value
    body = match.group("body")
    ref, kind = _split_kind(body)
    if kind is not None:
        return _resolve_tagged_reference(ref, kind)
    if "." in body:
        owner_ref, member = _split_owner(body)
        if member is None:
            ReferenceInvalid.report(f"Invalid reference '{body}'.")
            return None
        project, module, owner_name = _parse_reference(owner_ref)
        owner = _resolve_prototype(owner_name, project=project, module=module)
        return getattr(owner, member)
    project, module, name = _parse_reference(body)
    if name in builtins.__dict__:
        return builtins.__dict__[name]
    if name in NX_YAML_TYPES:
        return NX_YAML_TYPES[name]
    return _resolve_prototype(name, project=project, module=module)


def _construct_reference(loader, node):
    value = loader.construct_scalar(cast(yaml.ScalarNode, node))
    return _resolve_reference(value)


def _construct_prototype(loader, node):
    value = loader.construct_scalar(cast(yaml.ScalarNode, node))
    match = _REF_PATTERN.match(value)
    body = match.group("body") if match else value
    ref, kind = _split_kind(body)
    if kind is not None and kind != "prototype":
        ReferenceInvalid.report(f"Expected prototype reference, got '{value}'.")
        return None
    project, module, name = _parse_reference(ref)
    return _resolve_prototype(name, project=project, module=module)


def _construct_declarator(loader, node):
    value = loader.construct_scalar(cast(yaml.ScalarNode, node))
    match = _REF_PATTERN.match(value)
    body = match.group("body") if match else value
    ref, kind = _split_kind(body)
    if kind is None:
        kind = "field"
    return _resolve_tagged_reference(ref, kind)


def _construct_type(loader, node):
    value = loader.construct_scalar(cast(yaml.ScalarNode, node))
    match = _REF_PATTERN.match(value)
    body = match.group("body") if match else value
    ref, kind = _split_kind(body)
    if kind is None:
        kind = "type"
    return _resolve_tagged_reference(ref, kind)


def _register_aml_yaml_loader() -> None:
    nx_register_constructor("!Prototype", _construct_prototype)
    nx_register_constructor("!Declarator", _construct_declarator)
    nx_register_constructor("!Type", _construct_type)
    nx_register_constructor("tag:yaml.org,2002:str", _construct_reference)


_register_aml_yaml_loader()

# endregion

# =============================================================================
# Helpers
# =============================================================================
# region Helpers


def _is_docstring_expr(node: ast.AST) -> bool:
    """Return True if the node is a string literal expression (docstring)."""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
        node.value.value, str
    )


def _is_type_checking_guard(node: ast.AST) -> bool:
    """Return True for `if TYPE_CHECKING:` guards at module scope."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return isinstance(test.value, ast.Name) and test.value.id == "typing"
    return False


def _iter_allowed_body(
    body: list[ast.stmt],
    *,
    allowed: tuple[type[ast.AST], ...],
    class_allowed: tuple[type[ast.AST], ...],
) -> None:
    """Walk a statement body and enforce AML's allow list.

    This is a shallow enforcement on module and class bodies only. It does not
    inspect the content of function bodies or declarator callables.
    """
    for node in body:
        # Enforce statement type restrictions first.
        if not isinstance(node, allowed):
            DisallowedModuleStatement.report(
                f"AML disallows statement: {node.__class__.__name__}."
            )
            return
        # Only module docstrings are allowed as bare expressions.
        if isinstance(node, ast.Expr) and not _is_docstring_expr(node):
            DisallowedModuleStatement.report(
                "AML only allows docstring expressions at module scope."
            )
            continue
        # Allow TYPE_CHECKING guards and validate their nested bodies.
        if isinstance(node, ast.If):
            if not _is_type_checking_guard(node):
                DisallowedModuleStatement.report(
                    "AML only allows TYPE_CHECKING guards at module scope."
                )
                continue
            _iter_allowed_body(node.body, allowed=allowed, class_allowed=class_allowed)
            _iter_allowed_body(node.orelse, allowed=allowed, class_allowed=class_allowed)
        # Recurse into class bodies to validate class-level statements.
        if isinstance(node, ast.ClassDef):
            _iter_allowed_body(node.body, allowed=class_allowed, class_allowed=class_allowed)


def _validate_ast_allow_list(
    module_ast: ast.Module,
    *,
    module_allowed: tuple[type[ast.AST], ...],
    class_allowed: tuple[type[ast.AST], ...],
) -> None:
    """Validate module and class bodies against AML's allow list."""
    _iter_allowed_body(module_ast.body, allowed=module_allowed, class_allowed=class_allowed)


def _module_path(module: NxModuleType) -> Path:
    """Return the on-disk path for a module, raising on missing origins."""
    spec = module.__spec__
    origin = spec.origin if spec is not None else None
    origin = origin or getattr(module, "__file__", None)
    if not origin or origin == "built-in":
        ModuleSourceMissing.report(f"Module {module.__name__} has no file origin to parse.")
        return Path()
    return Path(origin)


def _ensure_module_ast(module: NxModuleType) -> ast.Module:
    """Return the module AST, parsing and caching it if needed."""
    module_ast = getattr(module, "__ast__", None)
    if isinstance(module_ast, ast.Module):
        return module_ast
    source = _module_path(module).read_text()
    # Parse and cache the module AST for later binding.
    module_ast = ast.parse(source)
    setattr(module, "__ast__", module_ast)
    return module_ast


def _name_assignments(node: ast.ClassDef) -> dict[str, ast.AST]:
    """Collect simple name assignments from a class body.

    This maps declarator names to their AST assignment or annotation nodes.
    """
    assignments: dict[str, ast.AST] = {}
    for subnode in node.body:
        match subnode:
            case ast.Assign():
                if len(subnode.targets) == 1:
                    target = subnode.targets[0]
                else:
                    continue
            case ast.AnnAssign():
                target = subnode.target
            case _:
                continue
        if isinstance(target, ast.Name):
            assignments[target.id] = subnode
    return assignments


def _collect_module_types(module: NxModuleType, module_ast: ast.Module) -> list[prototype]:
    """Collect prototype classes declared in the module.

    Only classes with a `prototype` metaclass and defined in this module are
    collected. Ordering follows the source order of class definitions.
    """
    class_defs = [node for node in module_ast.body if isinstance(node, ast.ClassDef)]
    class_names = {node.name for node in class_defs}
    prototypes = [
        value
        for name, value in module.__dict__.items()
        if isinstance(value, prototype)
        and name in class_names
        and value.__module__ == module.__name__
    ]
    order = {node.name: idx for idx, node in enumerate(class_defs)}
    prototypes.sort(key=lambda cls: order.get(cls.__name__, len(order)))
    # Cache on the module for downstream compiler usage.
    setattr(module, "__prototypes__", prototypes)
    return prototypes


def _assign_declarator_ast(cls: prototype, class_def: ast.ClassDef) -> None:
    """Bind AST assignment nodes to declarators declared on a prototype."""
    assignments = _name_assignments(class_def)
    for declarator in cls.__raw_declarators__.values():
        if declarator.__ast__ is not None:
            continue
        if (name := declarator.name) is None:
            continue
        if name in assignments:
            with diagnostic_context(declarator):
                declarator.__set_ast__(assignments[name])


def _backfill_annotation_types(cls: prototype, module: NxModuleType) -> None:
    """Resolve type hints and backfill declarator types.

    Forward references are evaluated with the module namespace, and type
    annotations are normalized into base types plus nullability and classvar
    flags.
    """
    try:
        hints = get_type_hints(
            cls,
            globalns={**module.__dict__, "ModuleDiagnosticBag": ModuleDiagnosticBag},
            localns={**module.__dict__, "ModuleDiagnosticBag": ModuleDiagnosticBag},
            include_extras=True,
        )
    except Exception as exc:  # noqa: BLE001
        TypeHintResolutionFailed.report(
            f"Failed to resolve type hints for {cls.__name__}: {exc}"
        )
        return
    for declarator in cls.__raw_declarators__.values():
        if not isinstance(declarator, AnnotatableDeclarator):
            continue
        if declarator.type is not None:
            continue
        if (name := declarator.name) is None or name not in hints:
            continue
        hint_value = hints[name]
        unwrap_classvar = getattr(cls, "__unwrap_classvar_type__", unwrap_classvar_type)
        unwrap_optional = getattr(cls, "__unwrap_optional_type__", unwrap_optional_type)
        hint_value, classvar = unwrap_classvar(hint_value)
        hint_value, nullable = unwrap_optional(hint_value)
        hint = hint_value if isinstance(hint_value, type) else None
        with diagnostic_context(declarator):
            declarator.__set_type__(
                declarator.annotation, hint, nullable, classvar, hint=hint_value
            )


def _validate_type_role(cls: prototype) -> None:
    """Validate archetype/trait/prototype role invariants."""
    if is_archetype(cls):
        if cls.__archetype__ is not cls:
            InvalidTypeRole.report(
                f"Archetype {cls.__name__} must reference itself as __archetype__."
            )
            return
        if cls.__role__ is not TypeRole.ARCHETYPE:
            InvalidTypeRole.report(f"Archetype {cls.__name__} must have role ARCHETYPE.")
            return
        return
    if is_trait(cls):
        if cls.__role__ is not TypeRole.TRAIT:
            InvalidTypeRole.report(f"Trait {cls.__name__} must have role TRAIT.")
            return
        return
    if is_prototype(cls):
        if cls.__role__ is not TypeRole.PROTOTYPE:
            InvalidTypeRole.report(f"Prototype {cls.__name__} must have role PROTOTYPE.")
            return
        # Pure prototypes cannot declare metadescriptors locally.
        metadescriptors = [
            registry
            for handle, registry in cls.__declarators__.items()
            if handle in ("metadata", "nxfield", "option", "metavalidator")
        ]
        if any(registry for registry in metadescriptors):
            PrototypeMetadescriptorForbidden.report(
                "Prototypes cannot declare metadescriptors."
            )
            return
        return
    InvalidTypeRole.report(
        f"AML type {cls.__name__} is not a valid archetype, trait, or prototype."
    )
    return


def _validate_declarators(cls: prototype) -> None:
    """Validate declarators against owner-aware rules."""
    # Defer to archetype and traits for domain-specific declarator validation.
    cls.__archetype__.__validate_declarators__(cls)
    for trait in cls.__traits__:
        trait.__validate_declarators__(cls)


def _validate_bindings(cls: prototype) -> None:
    """Validate bound values, attribute validators, and data parsers for a prototype."""
    declarators = cls.__merged_declarators__
    bindings = cls.__merged_bindings__
    # -------------------------
    # 1) Validate each bound value against its declarator contract.
    # -------------------------
    for binding_registry in (bindings.metadata, bindings.option):
        for name, value in binding_registry.items():
            declarator: Declarator = binding_registry.declarators[name]
            with diagnostic_context(declarator):
                is_valid = declarator.__validate_binding__(cls, value)
            if not is_valid:
                BindingInvalidValue.report(
                    (f"Binding '{name}' with value '{value}' is not valid "
                     f"for {declarator!r} in {cls!r}.")
                )
                continue
    # -------------------------
    # 2) Run attribute-level validators (metadata/option/nxfield validators).
    # -------------------------
    validators = declarators.metavalidator.values()
    attribute_validators = [v for v in validators if isinstance(v, EnumerationCallableDeclarator)]  # noqa
    for validator in attribute_validators:
        if is_not_missing(validator.callable):
            target_handle = validator.__handle__.removesuffix("_validator")
            binding_registry = bindings.get_registry(target_handle)
            for member in validator.members:
                if member in binding_registry:
                    value = binding_registry[member]
                    with diagnostic_context(validator):
                        is_valid = validator.callable(cls, value)
                    if not is_valid:
                        ValidatorFailed.report(
                            (f"Binding '{member}' with value '{value}' failed validation by "
                             f"{validator!r} in {cls!r}.")
                        )
                        continue
    # -------------------------
    # 3) Data parsing/validation is deferred to runtime.


def _validate_enumerations(cls: prototype) -> None:
    """Validate EnumerationDeclarator members against declared names."""
    # Enumerations reference members by name. We validate those names against
    # the merged declarator registries so inherited declarations are included.
    local_declarators = cls.__declarators__
    merged_declarators = cls.__merged_declarators__
    # We start with metavalidators, concretely meaning MetadataValidator,
    # NxFieldValidator, and OptionValidator declarators.
    metavalidators = local_declarators.metavalidator
    for declarator in metavalidators.values():
        if not isinstance(declarator, EnumerationCallableDeclarator):
            continue
        if not declarator.members:
            msg = (f"Enumeration declarator '{declarator.name}' in prototype "
                   f"'{cls.__name__}' must specify at least one member.")
            EnumerationInvalid.report(message=msg)
            continue
        target_handle = declarator.__handle__.removesuffix("_validator")
        target_registry = merged_declarators.get(target_handle, {})
        for member in declarator.members:
            if member not in target_registry:
                msg = (f"Enumeration member '{member}' not found for declarator "
                       f"'{declarator.name}' in prototype '{cls.__name__}'.")
                EnumerationInvalid.report(message=msg)
                continue
    # Next we validate any parsers or validators that are possible enumerations.
    constructor_registry = local_declarators.constructor
    for declarator in constructor_registry.values():
        declarator = cast(ParserDeclarator | ValidatorDeclarator, declarator)
        if issubclass(cls, Data):
            # Data prototypes cannot have enumeration parsers/validators.
            if declarator.members:
                msg = (f"{declarator} cannot have members in a Data prototype.")
                EnumerationInvalid.report(message=msg)
                continue
        elif issubclass(cls, Model):
            # Model prototypes implement both targeted and model-wide parsers/validators.
            if not declarator.members:
                # Declarator applies to the model itself, skip enumeration validation.
                continue
            else:
                # Declarator applies to specific attributes, validate members.
                target_registry = merged_declarators.field
                for member in declarator.members:
                    if member not in target_registry:
                        msg = (f"Enumeration member '{member}' not found for declarator "
                               f"'{declarator.name}' in prototype '{cls.__name__}'.")
                        EnumerationInvalid.report(message=msg)
                        continue
    # Next we validate field groups
    field_groups = [d for d in local_declarators.field.values() if isinstance(d, FieldGroupProtodescriptor)]  # noqa
    for field_group in field_groups:
        if not field_group.members:
            msg = (f"Field group '{field_group.name}' in prototype "
                   f"'{cls.__name__}' must specify at least one member.")
            EnumerationInvalid.report(message=msg)
            continue
        target_registry = merged_declarators.field
        for member in field_group.members:
            if member not in target_registry:
                msg = (f"Field group member '{member}' not found for declarator "
                       f"'{field_group.name}' in prototype '{cls.__name__}'.")
                EnumerationInvalid.report(message=msg)
                continue
    # Finally, we validate any schema declarator
    schema_declarators = [d for d in local_declarators.schema.values() if isinstance(d, EnumerationCallableDeclarator)]  # noqa
    for declarator in schema_declarators:
        if not declarator.members:
            msg = (f"Enumeration declarator '{declarator.name}' in prototype "
                   f"'{cls.__name__}' must specify at least one member.")
            EnumerationInvalid.report(message=msg)
            continue
        target_registry = merged_declarators.field
        for member in declarator.members:
            if member not in target_registry:
                msg = (f"Enumeration member '{member}' not found for declarator "
                       f"'{declarator.name}' in prototype '{cls.__name__}'.")
                EnumerationInvalid.report(message=msg)
                continue

# endregion

# =============================================================================
# Public API
# =============================================================================
# region Public API


def finalize_module(
    module: NxModuleType,
    *,
    strict: bool = True,
    module_allowed: tuple[type[ast.AST], ...] = DEFAULT_ALLOWED_MODULE_NODES,
    class_allowed: tuple[type[ast.AST], ...] = DEFAULT_ALLOWED_CLASS_NODES,
) -> None:
    """Finalize an AML module after import.

    Args:
        module: Imported AML module to finalize.
        strict: Whether to enforce AST statement allow-listing.
        module_allowed: Allowed AST node types at module scope.
        class_allowed: Allowed AST node types in class bodies.

    Raises:
        TypeError: If forward-reference resolution or AML validation fails.
        SyntaxError: If the module uses forbidden statements in strict mode.
        RuntimeError: If the module source cannot be located for parsing.
    """
    # Finalization is idempotent per module.
    if getattr(module, "__finalized__", False):
        return
    # set / reset a diagnostic bag.
    module.__diagnostics__ = ModuleDiagnosticBag(module)
    token = MODULE_DIAGNOSTICS.set(module.__diagnostics__)
    try:
        module_ast = _ensure_module_ast(module)
        if strict:
            # Enforce import safety and declarative-only module/class bodies.
            _validate_ast_allow_list(
                module_ast,
                module_allowed=module_allowed,
                class_allowed=class_allowed,
            )
        # Collect prototypes and bind their AST nodes.
        prototypes = _collect_module_types(module, module_ast)
        class_defs = {node.name: node for node in module_ast.body if isinstance(node, ast.ClassDef)}
        for cls in prototypes:
            with diagnostic_context(cls):
                if (class_def := class_defs.get(cls.__name__)) is not None:
                    cls.__ast__ = class_def
                    _assign_declarator_ast(cls, class_def)
                # Resolve forward references and validate type roles/enumerations.
                _backfill_annotation_types(cls, module)
                _validate_type_role(cls)
                _validate_declarators(cls)
                _validate_enumerations(cls)
        for cls in prototypes:
            with diagnostic_context(cls):
                _validate_bindings(cls)
                validators = cls.__merged_declarators__.metavalidator.values()
                prototype_validators = [v for v in validators if isinstance(v, PrototypeValidator)]
                for validator in prototype_validators:
                    if is_not_missing(validator.callable):
                        with diagnostic_context(validator):
                            is_valid = validator.callable(cls)
                        if not is_valid:
                            PrototypeValidatorFailed.report(
                                f"Prototype {cls!r} failed validation by {validator!r}."
                            )
                            continue
        PROTOTYPES.register_module(module, prototypes)
        # Mark module as finalized to avoid rework.
        setattr(module, "__finalized__", True)
        # Phase boundary: escalate any collected errors.
        raise_on_errors(module.__diagnostics__)
    finally:
        MODULE_DIAGNOSTICS.reset(token)

# endregion
