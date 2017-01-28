#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the DataTract class.

DataTract objects hold references to multiple classes that implement
different structures around a common data schema. These include
single record containers as well as containers for tabular data based
on pandas' DataFrame. The primary role of DataTract objects, besides holding
a Schema class, is to provide a namespace with consistent attribute names.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from functools import partial
import re
import sys

from ..utilities import nxattr, functions as fun
from ..meta.nxobject import NxObject
from ..meta.nxtype import nxtype, archetype
from ..meta.metadescriptors import TypeAttribute
from .exceptions import DataError
from . import schema as sch, record as rec, frame as frm

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
    The name is automatically derived from the Schema type's name, being
    either that same name, or the Schema's type name with 'Schema' truncated.
    """
    # Mapping of types to attribute names under a DataTract object
    __attributes__ = {sch.Schema: 'Schema',
                      rec.NxRecord: 'Record',
                      frm.NxDataCollection: 'Collection',
                      frm.NxDataSequence: 'Sequence',
                      frm.NxDataMapping: 'Mapping'}

    def __init__(self, schema, tbname=None):
        """Initializes a DataTract.

        attrs:
            schema: a Schema *class*
            tbname: an optional alternative name for database tables
                created from the Tract.
        """
        # First check that there is a proper schema
        if not issubclass(schema, sch.Schema):
            msg = "A tract must feature a Schema."
            raise TractDefinitionError(msg)
        self.Schema = schema
        schema.tract = self
        # Assembles name if needed
        try:
            self._name = re.match('\w+(?=Schema)', schema.__name__).group(0)
        except AttributeError:
            self._name = schema.__name__
        self._tbname = tbname or self.name
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
        return self.Collection.frametype


def tract(cls=None, *, tbname=None):
    """The tract decorator, which creates a DataTract from a Schema class.

    This decorator offers the most straightforward way to define a DataTract.
    The intended practice is to use the tract decorator on a Schema type
    declaration that does not include 'Schema' in its name, e.g.:

    @tract
    class User(sch.Schema):
        ...

    In which case the global name 'User' now refers to a DataTract, and
    the declared schema type is renamed 'UserSchema', and globally
    accessible as User.Schema.

    However as a convenience, the decorator also tolerates Schema
    declaration that explicilty add 'Schema' in the type name, in which
    case the tract receives that name with 'Schema' truncated (e.g.
    'UserSchema' makes a Tract named 'User'), and the tract's name is
    inserted into the global namespace of the module in which the Schema
    was declared.

    Params:
        cls: A Schema declaration.
        tbname: An alternative name for database tables created from the
            Tract.

    Raises:
        TractDefinitionError: if the supplied cls is not a subclass of Schema,
            or if its name doesn't follow conventions and no name is supplied.

    Returns:
        A DataTract instance.

    NOTE that care must be taken when defining DataTracts based
    on inherited schemas that have themselves been decorated. For instance,
    if Parent is a Schema type that was wrapped into a Tract, then
    a Child Tract would be declared as follows:

    @tract
    class Parent(sch.Schema):
        ...

    @tract
    class Child(Parent.Schema):
        ...

    """
    # Enables the decorator to function with or without arguments.
    if cls is None:
        return partial(tract, tbname=tbname)
    tract_ = DataTract(cls, tbname=tbname)
    try:
        name = re.match('\w+(?=Schema)', cls.__name__).group(0)
        sys.modules[cls.__module__].__dict__[name] = tract_
        return cls
    except AttributeError:
        cls.__name__ += 'Schema'
        cls.__qualname__ += 'Schema'
        return tract_

# =============================================================================
# DataTract class
# =============================================================================


@archetype
class DataChannel(NxObject):
    """An abstract base class for holding process-storage connections.

    A DataChannel requires a DataTract, and a store, which is any kind of
    object that sufficiently describes a data storage resource, e.g. a
    database table or a file.
    A DataChannel provides methods for loading / dumping data from / to
    a storage unit. Note that this is an abstract construct, i.e.
    instantiating a DataChannel does not actually create any kind of
    physical channel.
    The virtue of the DataChannel is to provide a unified interface between
    Anaximander DataObjects and persistent storage.
    """
    __loader__ = TypeAttribute(validate=fun.subcheck(frm.DataLoader))
    __dumper__ = TypeAttribute(validate=fun.subcheck(frm.DataDumper))

    def __init__(self, tract, store):
        self.tract = tract
        self.store = store

    @property
    def loader(self):
        return self.__loader__(self.schema, self.store)

    @property
    def dumper(self):
        return self.__dumper__(self.schema, self.store)

    @property
    def schema(self):
        return self.tract.Schema
