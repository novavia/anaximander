from typing import Any, Callable

import attrs

from .meta import DataobjectMetadescriptor


@attrs.define
class Field(DataobjectMetadescriptor):
    """Base class for field metadescriptors."""

    default: Any = attrs.field(default=None)
    factory: Callable | None = attrs.field(default=None)
    key: bool | str | list[str] | None = attrs.field(default=None)
    sequence: bool | str | list[str] | None = attrs.field(default=None)
    group: str | list[str] | None = attrs.field(default=None)
    index: bool | str | list[str] | None = attrs.field(default=None)
    unique: bool = attrs.field(default=False)
    repr: bool | Callable | None = attrs.field(default=None)


def field(
    *,
    default: Any = None,
    factory: Callable | None = None,
    key: bool | str | list[str] | None = None,
    sequence: bool | str | list[str] | None = None,
    group: str | list[str] | None = None,
    index: bool | str | list[str] | None = None,
    unique: bool = False,
    repr: bool | Callable | None = None,
) -> Any:
    """Function used in model declarations to define fields.

    Args:
        default (Any, optional): A default value when none is supplied. Defaults to None.
        factory (Callable | None, optional): A callable that generates default values. Defaults
            to None.
        key (bool | str | list[str] | None, optional): Flags the field as primary key, or
            part of the primary key, or belonging to a named key or list therof. Defaults to None.
        sequence (bool | str | list[str] | None, optional): Flags the field as a primary sequence,
            or part of the primary sequence, or belonging to a named sequence or list thereof.
            Defaults to None.
        group (str | list[str] | None, optional): Flags the field as belonging to a group or list
            thereof. Defaults to None.
        index (bool | str | list[str] | None, optional): Flags the field as being indexed or
            belonging to a named index or list thereof. Defaults to None.
        unique (bool, optional): Flags the field as having a unique value for each model instance
            that it describes. This is a syntactic shortcut for setting a unique constraint that
            applies to a single field. Defaults to False.
        repr (bool | Callable | None, optional): Flags whether the field should be included in
            model instance representations. Alternatively a callable may be supplied that
            transforms the field's value before passing it the instance representation method.
            If set to None, then built-in rules will determine whether the field is part of the
            model instance representation. Defaults to None.

    Returns:
        Any: A Field metadescriptor instance. Any is used as the return type to allow arbirary
            type annotations to be assigned to model attributes.
    """
    return Field(
        default=default,
        factory=factory,
        key=key,
        sequence=sequence,
        group=group,
        index=index,
        unique=unique,
        repr=repr,
    )


@attrs.define
class Relationship(DataobjectMetadescriptor):
    pass


@attrs.define
class Parent(Relationship):
    """Describes a parent relationship."""

    key: bool | str | list[str] | None = attrs.field(default=None)
    group: str | list[str] | None = attrs.field(default=None)
    index: bool | str | list[str] = attrs.field(default=True)
    unique: bool = attrs.field(default=False)
    repr: bool | Callable | None = attrs.field(default=None)


def parent(
    *,
    key: bool | str | list[str] | None = None,
    group: str | list[str] | None = None,
    index: bool | str | list[str] = True,
    unique: bool = False,
    repr: bool | Callable | None = None,
) -> Any:
    """Function used in model declarations to define a parent relationship.

    Args:
        key (bool | str | list[str] | None, optional): Flags the relationship as primary key, or
            part of the primary key, or belonging to a named key or list therof. Defaults to None.
        group (str | list[str] | None, optional): Flags the relationship as belonging to a group or
            list thereof. Defaults to None.
        index (bool | str | list[str], optional): Flags the relationship as being indexed or
            belonging to a named index or list thereof. Defaults to True.
        unique (bool, optional): Flags the relationship as having a unique value for each model
            instance that it describes. This is a syntactic shortcut for setting a unique
            constraint that applies to a single field or relationship. Defaults to False.
        repr (bool | Callable | None, optional): Flags whether the relationship should be included
            in model instance representations. Alternatively a callable may be supplied that
            transforms the relationship's value before passing it the instance representation
            method. If set to None, then built-in rules will determine whether the relationship is
            part of the model instance representation. Defaults to None.

    Returns:
        Any: A Parent metadescriptor instance. Any is used as the return type to allow arbirary
            type annotations to be assigned to model attributes.
    """
    return Parent(
        key=key,
        group=group,
        index=index,
        unique=unique,
        repr=repr,
    )


@attrs.define
class Query(Relationship):
    """Describes a query relationship."""

    pass


def query() -> Any:
    return Query()
