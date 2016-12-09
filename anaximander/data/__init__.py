#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This package defines Data, an abstraction that facilitate data handling.

A Data class serves first and foremost as a namespace for grouping
classes of data objects pertaining to a given type of data. For instance,
a class UserSession(Data) indicates that there is a type of data that
holds information about user sessions. That data may be accessed from files
or a database table, and consumed in tabular format using pandas dataframes
or as individual records. Anaximander offers to automate the creation of
classes to handle those use cases, and makes them accessible in the
UserSession namespace. Hence the programmer can bet on the existence of
classes UserSession.Schema, UserSession.Record, UserSession.Log, which
respectively provide access to a schema, a record class and a tabular data
class.

The purpose of the Data abstraction is to deal with information content
rather than typical application objects. Exemples of the latter would include
an Employee class in a payroll application or a Rectangle class in a plotting
framework. By contrast, Data could be subclassed into UserSession, which
is intended to capture historical information about actions taken by users of
a website, or Trajectory, which could serve to record time series of
moving object locations. There isn't a hard and fast rule that could
draw a clear distinction between the two, as this is purely a matter of
design. However, the primary purpose of this Data framework is to wrap
pandas dataframes with an interface that guarantees column names and types.
Hence it is intended for dealing with data that is most typically consumed
in tabular form -think logs as the most obvious exemple. Individual rows
can be pulled as 'Records' and these are objects, but as the name implies
they are thought of as mere data containers rather than fancy objects. That
being said, nothing prevents a programmer from adding attributes and methods
to an automatically generated record class. Further, by wrapping dataframes
into container objects that make interface guarantees, the Anaximander data
framework actually allows a much more object-oriented treatment of these
dataframes. For instance, a UserSessionLog class can define various
methods that are specific to the type of data it contains, something that
is not as elegantly possible with raw dataframes -i.e. subclassing is brittle
and the alternative is to resort to externally defined functions, losing
the benefits of object-oriented programming. Indeed the best justification
for the data package is probably the elevation of pandas' dataframe to the
role of a structural template that serves to define semantic hierarchies
of specialized data containers following an object-oriented paradigm; in
other words, a marriage between 'data' and 'object' -hence the DataObject.

Because the definition of a Data class hinges primarily on a schema,
the canonical mechanism for creating it is to decorate a Schema declaration.
Alternatively, the Schema declaration can be embedded in a Data class
declaration, or the two declarations can be written independently as long
as a reference to the Schema class is supplied to the Data class.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import abc
from functools import partial
import re
import types

import attr

from . import schema as sch, record as rec
from ..utilities.xprops import cachedproperty, weakproperty

# =============================================================================
# Tract metaclass
# =============================================================================


class DataError(Exception):
    """Base exception type for Data-related errors."""
    pass


class DataDefinitionError(DataError):
    """Raised when the definition of a DataType fails."""
    pass


class DataType(abc.ABCMeta):
    """The metaclass for Data classes."""

    def __init__(cls, name, bases, attrs):
        # First check that there is a proper schema
        try:
            assert issubclass(cls.Schema, sch.Schema)
        except (AttributeError, AssertionError):
            msg = "A Data class must feature a Schema."
            raise DataDefinitionError(msg)

        # Then verify consistent bases for Data and Schema
        if name != 'Data':
            base_datas = cls.base_datas
            base_schemas = cls.Schema.base_schemas
            try:
                assert [d.Schema for d in base_datas] == list(base_schemas)
            except AssertionError:
                msg = "Inconsistent bases for {0} and {1}"
                raise DataDefinitionError(msg.format(cls, cls.Schema))

    @property
    def base_datas(cls):
        """Filters bases for Data subclasses."""
        return tuple(c for c in cls.__bases__ if issubclass(c, Data))

    @classmethod
    def make_record_attributes(mcl, schema):
        """Extract a list of attribute specifications from Schema class."""
        attrs = {}
        for k, v in schema.fields.items():
            if v.required:
                attrs[k] = attr.ib()
            else:
                attrs[k] = attr.ib(default=v._attribute_default())
        # Special attribute _validate makes it possible to add a 'validate'
        # option to the __init__ method of the record class, while ignoring
        # it for most practical purposes.
        validate = attr.ib(False, repr=False, cmp=False, hash=False)
        attrs['_validate'] = validate
        return attrs

    @classmethod
    def make_record_class(mcl, cls, name=None):
        """Creates a record class to assign to cls."""
        name = name or cls.__name__ + 'Record'

        def body(ns):
            """Populates the class' namespace with field attributes."""
            attributes = mcl.make_record_attributes(cls.Schema)
            ns.update(attributes)

        bases = tuple(b.Record for b in cls.base_datas) or (rec.Record,)
        kls = types.new_class(name, bases, exec_body=body)
        record_class = attr.s(kls)
        return record_class

    def set_record_class(cls, record_class):
        """Sets the record class."""
        if not issubclass(record_class, rec.Record):
            raise DataDefinitionError()
        cls.Record = record_class
        record_class.Data = cls


class Data(metaclass=DataType):
    """The base Data class."""

    class Schema(sch.Schema):
        pass


# XXX: how is inheritance possible? Is it at all desirable? Certainly 
# we want it for the Frame.
# Is is enabled through Schema, through Data, through both, neither?
def data(cls=None, *, name=None):
    """The data decorator, which turns Schema classes into Data classes.

    This decorator offers the most straightforward way to define a Data
    class. Of course, the fact that it returns an entirely different
    class than the one being decorated is sure to raise eyebrows, and
    its use is not mandatory.

    Params:
        cls: A Schema declaration.
        name: An optional name for the Data class. If not supplied,
            the function expects that the schema class has a name
            in the form [CamelCase]Schema and will extract CamelCase as the
            name of the Data class.

    Raises:
        DataDefinitionError: if the supplied cls is not a subclass of Schema,
            or if its name doesn't follow conventions and no name is supplied.

    Returns:
        A Data class, with a Schema attribute that is the decorated class.
    """
    # Enables the decorator to function with or without arguments.
    if cls is None:
        return partial(data, name=name)

    if name is not None:
        try:
            name = re.match('\w+(?=Schema)', cls.__name__).group(0)
        except AttributeError:
            msg = "The schema has a non-standard name. Either change " + \
                "its name to end with 'Schema' or supply a custom name " + \
                "for the target Data class."
            raise DataDefinitionError(msg)

    # Extracts relevant bases
    base_datas = (bs.data for bs in cls.base_schemas if bs.data is not None)
    bases = tuple(base_datas) or (Data,)

    attrs = {'Schema': cls}

    # Return class
    rcls = DataType(name, bases, attrs)
    cls.data = rcls  # Assigns the data class to the schema class

    return rcls
