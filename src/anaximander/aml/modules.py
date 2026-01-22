"""Finalize AML modules after import."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import ast
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, get_args, get_origin, get_type_hints

from .declarative import AnnotatableDeclarator
from .prototype import TypeRole, is_archetype, is_prototype, is_trait, prototype

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
# Helpers
# =============================================================================
# region Helpers


def _is_docstring_expr(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
        node.value.value, str
    )


def _is_type_checking_guard(node: ast.AST) -> bool:
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
    for node in body:
        if not isinstance(node, allowed):
            raise SyntaxError(f"AML disallows statement: {node.__class__.__name__}.")
        if isinstance(node, ast.Expr) and not _is_docstring_expr(node):
            raise SyntaxError("AML only allows docstring expressions at module scope.")
        if isinstance(node, ast.If):
            if not _is_type_checking_guard(node):
                raise SyntaxError("AML only allows TYPE_CHECKING guards at module scope.")
            _iter_allowed_body(node.body, allowed=allowed, class_allowed=class_allowed)
            _iter_allowed_body(node.orelse, allowed=allowed, class_allowed=class_allowed)
        if isinstance(node, ast.ClassDef):
            _iter_allowed_body(node.body, allowed=class_allowed, class_allowed=class_allowed)


def _validate_ast_whitelist(
    module_ast: ast.Module,
    *,
    module_allowed: tuple[type[ast.AST], ...],
    class_allowed: tuple[type[ast.AST], ...],
) -> None:
    _iter_allowed_body(module_ast.body, allowed=module_allowed, class_allowed=class_allowed)


def _module_path(module: ModuleType) -> Path:
    spec = module.__spec__
    origin = spec.origin if spec is not None else None
    origin = origin or getattr(module, "__file__", None)
    if not origin or origin == "built-in":
        raise RuntimeError(f"Module {module.__name__} has no file origin to parse.")
    return Path(origin)


def _ensure_module_ast(module: ModuleType) -> ast.Module:
    module_ast = getattr(module, "__ast__", None)
    if isinstance(module_ast, ast.Module):
        return module_ast
    source = _module_path(module).read_text()
    module_ast = ast.parse(source)
    setattr(module, "__ast__", module_ast)
    return module_ast


def _name_assignments(node: ast.ClassDef) -> dict[str, ast.AST]:
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


def _collect_module_types(module: ModuleType, module_ast: ast.Module) -> list[prototype]:
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
    setattr(module, "__types__", prototypes)
    return prototypes


def _assign_declarator_ast(cls: prototype, class_def: ast.ClassDef) -> None:
    assignments = _name_assignments(class_def)
    for declarator in cls.__declarations__.values():
        if declarator.__ast__ is not None:
            continue
        if (name := declarator.name) is None:
            continue
        if name in assignments:
            declarator.__set_ast__(assignments[name])


def _unwrap_optional_type(type_hint: Any) -> tuple[Any, bool]:
    if type_hint is None:
        return None, False
    args = get_args(type_hint)
    if args and any(arg is type(None) for arg in args):
        non_none_args = [arg for arg in args if arg is not type(None)]
        base_type = non_none_args[0] if non_none_args else None
        return base_type, True
    return type_hint, False


def _unwrap_classvar_type(type_hint: Any) -> tuple[Any, bool]:
    if get_origin(type_hint) is ClassVar:
        args = get_args(type_hint)
        base_type = args[0] if args else None
        return base_type, True
    return type_hint, False


def _backfill_annotation_types(cls: prototype, module: ModuleType) -> None:
    try:
        hints = get_type_hints(
            cls, globalns=module.__dict__, localns=module.__dict__, include_extras=True
        )
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"Failed to resolve type hints for {cls.__name__}: {exc}") from exc
    for declarator in cls.__declarations__.values():
        if not isinstance(declarator, AnnotatableDeclarator):
            continue
        if declarator.type is not None:
            continue
        if (name := declarator.name) is None or name not in hints:
            continue
        hint_value = hints[name]
        unwrap_classvar = getattr(cls, "__unwrap_classvar_type__", _unwrap_classvar_type)
        unwrap_optional = getattr(cls, "__unwrap_optional_type__", _unwrap_optional_type)
        hint_value, classvar = unwrap_classvar(hint_value)
        hint_value, nullable = unwrap_optional(hint_value)
        hint = hint_value if isinstance(hint_value, type) else None
        declarator.__set_type__(declarator.annotation, hint, nullable, classvar)


def _validate_type_role(cls: prototype) -> None:
    if is_archetype(cls):
        if cls.__archetype__ is not cls:
            raise TypeError(f"Archetype {cls.__name__} must reference itself as __archetype__.")
        if cls.__role__ is not TypeRole.ARCHETYPE:
            raise TypeError(f"Archetype {cls.__name__} must have role ARCHETYPE.")
        return
    if is_trait(cls):
        if cls.__role__ is not TypeRole.TRAIT:
            raise TypeError(f"Trait {cls.__name__} must have role TRAIT.")
        return
    if is_prototype(cls):
        if cls.__role__ is not TypeRole.PROTOTYPE:
            raise TypeError(f"Prototype {cls.__name__} must have role PROTOTYPE.")
        return
    raise TypeError(f"AML type {cls.__name__} is not a valid archetype, trait, or prototype.")

# endregion

# =============================================================================
# Public API
# =============================================================================
# region Public API


def finalize_module(
    module: ModuleType,
    *,
    strict: bool = True,
    module_allowed: tuple[type[ast.AST], ...] = DEFAULT_ALLOWED_MODULE_NODES,
    class_allowed: tuple[type[ast.AST], ...] = DEFAULT_ALLOWED_CLASS_NODES,
) -> None:
    """Finalize an AML module after import.

    Args:
        module: Imported AML module to finalize.
        strict: Whether to enforce AST statement whitelisting.
        module_allowed: Allowed AST node types at module scope.
        class_allowed: Allowed AST node types in class bodies.

    Raises:
        TypeError: If forward-reference resolution or AML validation fails.
        SyntaxError: If the module uses forbidden statements in strict mode.
        RuntimeError: If the module source cannot be located for parsing.
    """
    if getattr(module, "__finalized__", False):
        return
    module_ast = _ensure_module_ast(module)
    if strict:
        _validate_ast_whitelist(module_ast, module_allowed=module_allowed, class_allowed=class_allowed)
    prototypes = _collect_module_types(module, module_ast)
    class_defs = {node.name: node for node in module_ast.body if isinstance(node, ast.ClassDef)}
    for cls in prototypes:
        if (class_def := class_defs.get(cls.__name__)) is not None:
            cls.__ast__ = class_def
            _assign_declarator_ast(cls, class_def)
        _backfill_annotation_types(cls, module)
        _validate_type_role(cls)
    setattr(module, "__finalized__", True)

# endregion
