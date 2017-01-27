#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the DataTract class.

DataTract objects hold references to multiple classes that implement
different structures around a common data schema. These include
single record containers as well as containers for tabular data based
on pandas' DataFrame. The primary role of DataTract objects is to provide
a namespace with consistent attributes.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from functools import partial
import re
import sys

from .exceptions import DataError
from . import schema as sch, record as rec, frame as frm
from ..meta.nxtype import nxtype

# =============================================================================
# DataTract class
# =============================================================================


class TractError(DataError):
    """Base exception type for DataTract-related errors."""
    pass


class TractDefinitionError(TractError):
    """Raised when the definition of a DataTract fails."""
    pass


class DataTract:
    """A container for a Schema and related DataObject classes.

    By convention, a tract's name should be camelcase, as if it was a class.
    This is justified by the fact that the tract primarily serves as
    a namespace to access classes (e.g. Production.Record, Production.Log).
    """
    # Mapping of types to attribute names under a DataTract object
    __attributes__ = {sch.Schema: 'Schema',
                      rec.NxRecord: 'Record',
                      frm.NxDataCollection: 'Collection',
                      frm.NxDataSequence: 'Sequence',
                      frm.NxDataMapping: 'Mapping'}

    def __init__(self, schema, name=None, tbname=None):
        """Initializes a DataTract.

        attrs:
            schema: a Schema *class*
            name (str): an optional name for self, which is inserted into
                its module's global namespace. If no name is supplied, it
                is assembled automatically from the Schema class name,
                provided it follows the naming convention [CamelCase]Schema.
        """
        # First check that there is a proper schema
        if not issubclass(schema, sch.Schema):
            msg = "A tract must feature a Schema."
            raise TractDefinitionError(msg)
        self.Schema = schema
        schema.tract = self

        # Assembles name if needed
        if name is None:
            try:
                name = re.match('\w+(?=Schema)', schema.__name__).group(0)
            except AttributeError:
                msg = "The schema has a non-standard name. Either change " + \
                    "its name to end with 'Schema' or supply a custom " + \
                    "name for the target Data class."
                raise TractDefinitionError(msg)
        self._name = name
        self._tbname = tbname or name
        # Insert self in the schema's global namespace with name
        sys.modules[schema.__module__].__dict__[name] = self

        # Makes the record class
        self._make_record_class()

    @property
    def name(self):
        return self._name

    @property
    def tbname(self):
        """A name for tables created from a Tract."""
        return self._tbname

    @property
    def base(self):
        """Returns a 'base' tract from Schema's bases, or None."""
        base_schemas = self.Schema.base_schemas
        try:
            return base_schemas[0].tract
        except IndexError:
            return None

    def _basetype(self, type_):
        """Returns the base class for a given DataObject type.

        attrs:
            type_: a DataObject base type such as NxFrame or NxRecord.

        returns:
            a base class for the passed object type.
        """
        name = self.__attributes__[type_]
        try:
            return getattr(self.base, name, type_)
        except AttributeError:
            return type_

    def _make_record_class(self):
        """Creates a basic record class based on self's Schema."""
        name = self.name + 'Record'
        base = self._basetype(rec.NxRecord)
        nxtype(base, name=name, schema=self.Schema)

    @property
    def Record(self):
        return rec.NxRecord[self.Schema]

    @property
    def Collection(self):
        return frm.NxDataCollection[self.Schema]

    @property
    def Sequence(self):
        return frm.NxDataSequence[self.Schema]

    @property
    def Mapping(self):
        return frm.NxDataMapping[self.Schema]

    @property
    def Frame(self):
        """Returns the default NxDataFrame subtype for arbitrary data sets.

        If self's schema contains non-sequential keys, it is a Mapping.
        Else if there is a sequential key it is a Sequence.
        Otherwise it is a Collection.
        """
        if self.Schema.nskeys:
            return frm.NxDataMapping[self.Schema]
        elif self.Schema.seqkey:
            return frm.NxDataSequence[self.Schema]
        else:
            return frm.NxDataCollection[self.Schema]


def tract(cls=None, *, name=None, tbname=None):
    """The tract decorator, which picks up a DataTract from a Schema class.

    This decorator offers the most straightforward way to define a DataTract.
    Params:
        cls: A Schema declaration.
        name: An optional name for the DataTract. If not supplied,
            the function expects that the schema class has a name
            in the form [CamelCase]Schema and will extract CamelCase as the
            name of the DataTract.

    Raises:
        TractDefinitionError: if the supplied cls is not a subclass of Schema,
            or if its name doesn't follow conventions and no name is supplied.

    Returns:
        The decorated class.
    """
    # Enables the decorator to function with or without arguments.
    if cls is None:
        return partial(tract, name=name, tbname=tbname)
    DataTract(cls, name=name, tbname=tbname)
    return cls
