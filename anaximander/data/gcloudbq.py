#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides an interface from the data package to Google BigQuery.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import re

from googleapiclient.errors import HttpError
from gcloud import bigquery as bq

from ..utilities import functions as fun

# =============================================================================
# Custom exceptions
# =============================================================================


# XXX: not sure about keeping.
class _HttpErrorPatch:
    """A patch to extract messages from HttpErrors."""

    @property
    def message(self):
        string = self.content.decode()
        match = re.search('(?<="message": )".*"', string)
        if match is None:
            msg = "Could not extract message from {0}"
            raise AttributeError(msg.format(self))
        return match.group(0).strip('"')

fun.monkeypatch(HttpError, _HttpErrorPatch)


class BigQueryException(Exception):
    """Exception related to BigQuery administration."""
    pass

# =============================================================================
# Field mapping
# =============================================================================


_field_map = {'Field': 'STRING',
              'Raw': 'STRING',
              'Nested': 'RECORD',
              'Dict': NotImplemented,
              'List': NotImplemented,
              'String': 'STRING',
              'UUID': 'STRING',
              'Number': 'FLOAT',
              'Integer': 'INTEGER',
              'Decimal': 'FLOAT',
              'Boolean': 'BOOLEAN',
              'FormattedString': 'STRING',
              'Float': 'FLOAT',
              'DateTime': 'TIMESTAMP',
              'LocalDateTime': 'TIMESTAMP',
              'Time': 'TIME',
              'Date': 'DATE',
              'TimeDelta': 'INTEGER',
              'Url': 'STRING',
              'URL': 'STRING',
              'Email': 'STRING',
              'Method': NotImplemented,
              'Function': NotImplemented,
              'Str': NotImplemented,
              'Bool': NotImplemented,
              'Int': NotImplemented,
              'Constant': NotImplemented,
              }


def _field_type(field):
    """Returns a BiqQuery field type from a Schema field or NotImpemented."""
    return _field_map.get(type(field).__name__, NotImplemented)


def bqfield(field):
    """Returns a BigQuery schema field from a Field instance.

    Args:
        field (obj): a Field instance.

    Returns:
        a BigQuery SchemaField instance.
    """
    name = field.name
    if name is None:
        msg = "Field names must be set in their metadata attributes " + \
            "in order to be usable in BigQuery schemas."
        raise ValueError(msg)
    field_type = _field_type(field)
    if field_type is NotImplemented:
        msg = "Cannot use field of type {0} in BigQuery schema."
        raise ValueError(msg.format(type(field).__name__))
    mode = 'REQUIRED' if field.required else 'NULLABLE'
    description = field.description
    try:
        # Serializes a nested field
        fields = bqschema(field.nested)
    except AttributeError:
        fields = None
    return bq.SchemaField(name, field_type, mode,
                          description=description, fields=fields)

# =============================================================================
# Schema specification and table creation
# =============================================================================


def bqschema(schema):
    """Makes a BiqQuery schema from a Schema instance.

    Args:
        schema (obj): a Schema instance.

    Returns:
        list of SchemaField instances.
    """
    try:
        return [bqfield(f) for f in schema.fields.values()]
    except ValueError:
        raise ValueError("Schema cannot be serialized.")
