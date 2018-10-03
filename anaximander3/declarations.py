#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides schema decorators to set up io and transforms.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements and constants
# =============================================================================

from functools import partial

from .io.store import Title, Tract
from .transforms.tasks import insert_task_factory

__all__ = []

# =============================================================================
# Decorators
# =============================================================================


def _title_init(self):
    for store in self.stores:
        _deferred_tract(store, self)
    if self.pipeline is not None:
        pipeline(_title=self, **self.pipeline)
    if self.buffer is not None:
        buffer(_title=self, **self.buffer)
    if self.archive is not None:
        archive(_title=self, **self.archive)

Title._init_patch = _title_init


def _deferred_tract(role, title=None, schema=None, name=None, **kwargs):
    if title is None:
        if name is None:
            if not isinstance(schema, type):
                schema = type(schema)
            name = getattr(schema, '__title__', schema.__name__)
        title = Title(name, schema)
    Tract._deferred[role][title] = kwargs
    return title


def pipeline(schema=None, *, name=None, source=None, max_latency=None,
             _title=None, **kwargs):
    if _title is None:
        if schema is None:
            return partial(archive, name=name, **kwargs)
    title = _deferred_tract('pipeline', title=_title, schema=schema,
                            name=name, **kwargs)
    if source is not None:
        locals_ = locals()
        task_kwargs = {k: locals_[k] for k in ['max_latency']}
        task_kwargs = {k: v for k, v in task_kwargs.items() if v is not None}
        insert_task_factory(title, source, **task_kwargs)
    return schema


def buffer(schema=None, *, name=None, depth=None, _title=None, **kwargs):
    if _title is None:
        if schema is None:
            return partial(archive, name=name, depth=depth, **kwargs)
    _deferred_tract('buffer', title=_title, schema=schema, name=name,
                    depth=depth, **kwargs)
    return schema


def archive(schema=None, *, name=None, _title=None, **kwargs):
    if _title is None:
        if schema is None:
            return partial(archive, name=name, **kwargs)
    _deferred_tract('archive', title=_title, schema=schema, name=name,
                    **kwargs)
    return schema


def storage(schema=None, *, name=None, **kwargs):
    if schema is None:
        return partial(storage, name=name, **kwargs)
    pipeline(schema, name=name, **kwargs)
    buffer(schema, name=name, **kwargs)
    archive(schema, name=name, **kwargs)
