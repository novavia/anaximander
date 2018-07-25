#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the schema base types.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
from collections import Mapping, OrderedDict, defaultdict
import itertools
import types

import pandas as pd

from ..utilities import functions as fun, nxtime
from ..utilities.jsonmixin import jsonio
from .exceptions import SchemaError
from .nxcolumns import NxColumn, DateTime, TimeDelta, String, Categorical, \
    EventLabel, StateLabel, SessionLabel


__all__ = ['Schema', 'MultiSchema', 'LogsSchema', 'SampleLogsSchema',
           'EventLogsSchema', 'StateLogsSchema', 'SessionLogsSchema',
           'PeriodLogsSchema']

# =============================================================================
# Base types
# =============================================================================


class ColumnMapType(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        NxColumn.collect(cls, namespace)

    @property
    def index_columns(cls):
        return OrderedDict(fun.vfilter(lambda f: f.index, cls.__nxcolumns__))

    @property
    def sequencer_columns(cls):
        return OrderedDict(fun.vfilter(lambda f: f.index == 'sequential',
                                       cls.__nxcolumns__))

    @property
    def identifier_columns(cls):
        return OrderedDict(fun.vfilter(lambda f: f.index == 'nominal',
                                       cls.__nxcolumns__))

    @property
    def suffix_columns(cls):
        return OrderedDict(fun.vfilter(lambda f: f.index == 'suffix',
                                       cls.__nxcolumns__))

    @property
    def required_columns(cls):
        return OrderedDict(fun.vfilter(lambda f: f.required,
                                       cls.__nxcolumns__))

    @property
    def generic_columns(cls):
        return OrderedDict(fun.vfilter(lambda f: f.generic,
                                       cls.__nxcolumns__))

    @property
    def payload_columns(cls):
        columns = OrderedDict()
        for k, v in cls.__nxcolumns__.items():
            if isinstance(v, ColumnFamily):
                continue
            if not v.index:
                columns[k] = v
        return columns

    @property
    def column_families(cls):
        return OrderedDict(fun.vfilter(fun.typecheck(ColumnFamily),
                                       cls.__nxcolumns__))


class ColumnMap(Mapping, metaclass=ColumnMapType):
    """An ordered mapping of names to columns.

    Params:
        *columns: column names declared in the type to include in the object.
            If empty, then all columns are included. Passing unrecognized names
            will raise a SchemaError.
        exclude: conversely, it is possible to pass an iterable of excluded
            columns.
    Raises:
        SchemaError if unknown column names are used or if a required column
        is excluded.
    """

    def __init__(self, *columns, exclude=None):
        coldict = OrderedDict()
        colnames = set(self.__nxcolumns__)
        if columns:
            include = set(columns)
        else:
            include = colnames - set(type(self).generic_columns)
        exclude = set(fun.get(exclude, set()))
        include = include - exclude

        if any(n not in include for n in type(self).required_columns):
            msg = f"Cannot instantiate column map without required " + \
                  f"fields {type(self).required_columns}"
            raise SchemaError(msg)

        roster = itertools.chain(columns, self.__nxcolumns__)
        for name in roster:
            if name not in include or name in coldict:
                continue
            if name not in colnames:
                try:
                    key, suffix = name.split('#')
                    gcol = self.__nxcolumns__[key]
                    assert gcol.generic is True
                except (ValueError, KeyError, AssertionError):
                    msg = f"Unrecognized column name {name} passed " + \
                          f"to {type(self)}."
                    raise SchemaError(msg)
                else:
                    col = gcol.copy()
                    col.bind(name, col.cls)
                    col.generic = gcol
                    coldict[suffix] = col
            else:
                col = self.__nxcolumns__[name]
                if col.generic is True:
                    msg = f"Cannot instantiate column map with generic column."
                    raise SchemaError(msg)
                coldict[name] = self.__nxcolumns__[name]

        self._columns = coldict
        for k, v in self._columns.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return self._columns[key]

    def __iter__(self):
        return self._columns.__iter__()

    def __len__(self):
        return self._columns.__len__()

    def __hash__(self):
        return hash(tuple(self.values()))

    def __eq__(self, other):
        return tuple(self.values()) == tuple(other.values())

    def __repr__(self):
        typename = type(self).__name__
        colnames = list(self._columns)
        return f"<{typename} {colnames}>"


class SchemaIndexType(ColumnMapType):
    """Metaclass for SchemaIndex."""

    def __new__(mcl, name, bases, namespace):
        if bases[0] == ColumnMap:
            return super().__new__(mcl, name, bases, namespace)
        cmap = ColumnMapType(name, (), namespace)
        identifier = bool(cmap.identifier_columns)
        sequencer = bool(cmap.sequencer_columns)
        suffix = bool(cmap.suffix_columns)
        if identifier and sequencer:
            if suffix:
                metaclass = SuffixedDualSchemaIndexType
            else:
                metaclass = DualSchemaIndexType
        elif identifier:
            metaclass = NominalSchemaIndexType
        elif sequencer:
            metaclass = SequentialSchemaIndexType
        else:
            metaclass = EmptySchemaIndexType
        if not issubclass(metaclass, mcl):
            msg = f"Cannot subclass {bases[0].__name__} into a different " + \
                  f"index type {metaclass.__name__}"
            raise SchemaError(msg)
        return super().__new__(metaclass, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if len(cls.identifier_columns) > 1:
            msg = "SchemaIndex classes feature at most one identifier column."
            raise SchemaError(msg)
        if len(cls.sequencer_columns) > 1:
            msg = "SchemaIndex classes feature at most one sequencer column."
            raise SchemaError(msg)
        if len(cls.suffix_columns) > 1:
            msg = "SchemaIndex classes feature at most one suffix column."
            raise SchemaError(msg)
        cls.rowkey = cls.__rowkey__
        cls.rowidx = cls.__rowidx__


class EmptySchemaIndexType(SchemaIndexType):
    """A schema index with no columns."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        try:
            assert not cls.identifier_columns
            assert not cls.sequencer_columns
            assert not cls.suffix_columns
        except AssertionError:
            msg = "Incorrect index columns specification."
            raise SchemaError(msg)
        else:
            cls._identifer = None
            cls._sequencer = None
            cls._suffix = None


class NominalSchemaIndexType(SchemaIndexType):
    """A schema index with a single, nominal column."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        try:
            assert cls.identifier_columns
            assert not cls.sequencer_columns
            assert not cls.suffix_columns
        except AssertionError:
            msg = "Incorrect index columns specification."
            raise SchemaError(msg)
        else:
            cls._identifier = list(cls.identifier_columns)[0]
            cls._sequencer = None
            cls._suffix = None


class SequentialSchemaIndexType(SchemaIndexType):
    """A schema index with a single, sequential column."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        try:
            assert not cls.identifier_columns
            assert cls.sequencer_columns
            assert not cls.suffix_columns
        except AssertionError:
            msg = "Incorrect index columns specification."
            raise SchemaError(msg)
        else:
            cls._identifier = None
            cls._sequencer = list(cls.sequencer_columns)[0]
            cls._suffix = None


class DualSchemaIndexType(SchemaIndexType):
    """A schema index with a nominal and a sequential column."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        try:
            assert cls.identifier_columns
            assert cls.sequencer_columns
            assert not cls.suffix_columns
        except AssertionError:
            msg = "Incorrect index columns specification."
            raise SchemaError(msg)
        else:
            cls._identifier = list(cls.identifier_columns)[0]
            cls._sequencer = list(cls.sequencer_columns)[0]
            cls._suffix = None


class SuffixedDualSchemaIndexType(SchemaIndexType):
    """A schema index with a nominal and a sequential column."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        try:
            assert cls.identifier_columns
            assert cls.sequencer_columns
            assert cls.suffix_columns
        except AssertionError:
            msg = "Incorrect index columns specification."
            raise SchemaError(msg)
        else:
            cls._identifier = list(cls.identifier_columns)[0]
            cls._sequencer = list(cls.sequencer_columns)[0]
            cls._suffix = list(cls.suffix_columns)[0]


class SchemaIndex(ColumnMap, metaclass=SchemaIndexType):
    """Base class for SchemaIndex objects."""
    _identifier = None
    _sequencer = None
    _suffix = None

    def __init__(self):
        super().__init__()

    @property
    def sequencer(self):
        return self._sequencer

    @property
    def identifier(self):
        return self._identifier

    @property
    def suffix(self):
        return self._suffix

    @abc.abstractmethod
    def __rowkey__(self, index):
        """Method to compute the row key from index tuple."""
        return ""

    @abc.abstractmethod
    def __rowidx__(self, key):
        """Method to compute index columns from key."""
        return ()


class ColumnFamily(ColumnMap, NxColumn):
    """A column map that also serves as a data column in a larger schema.

    This is designed to map to column families in columnar databases.
    """

    def __init__(self, *columns, required=False, exclude=None):
        NxColumn.__init__(self, index=None, required=required)
        super().__init__(*columns, exclude=exclude)


class SchemaBaseType(ColumnMapType):
    __registry__ = dict()

    def __new__(mcl, name, bases, namespace, index=None):
        return super().__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace, index=None):
        super().__init__(name, bases, namespace)
        base_index = fun.get(index, getattr(cls, '__index__', SchemaIndex))
        ixcols = any(c.cls is cls for c in cls.index_columns.values())
        keydef = any(a in namespace for a in ['__rowkey__', '__rowidx__'])
        if ixcols or keydef:
            ix_name = name + 'Index'
            ix_namespace = cls.index_columns
            if hasattr(cls, '__rowkey__'):
                ix_namespace['__rowkey__'] = cls.__rowkey__
            if hasattr(cls, '__rowidx__'):
                ix_namespace['__rowidx__'] = cls.__rowidx__

            def exec_body(ns):
                ns.update(ix_namespace)
                return ns

            cls.__index__ = types.new_class(ix_name, (base_index,),
                                            exec_body=exec_body)
        else:
            cls.__index__ = base_index
            nxcolumns = base_index.__nxcolumns__.copy()
            nxcolumns.update(cls.__nxcolumns__)
            cls.__nxcolumns__ = nxcolumns


class SchemaBase(ColumnMap, metaclass=SchemaBaseType):
    """Base class for schemas."""

    def __init__(self):
        self.index = self.__index__()
        self._columns = OrderedDict(self.index._columns)

    @property
    def sequencer(self):
        return self.index._sequencer

    @property
    def identifier(self):
        return self.index._identifier

    @property
    def suffix(self):
        return self.index._suffix

    def rowkey(self, idx):
        """Method to compute the row key from index tuple."""
        return self.index.rowkey(idx)

    def rowidx(self, key):
        """Method to compute index columns from key."""
        return self.index.rowidx(key)


class SchemaType(SchemaBaseType):

    def __init__(cls, name, bases, namespace, index=None, payload=None):
        if cls.regkey in cls.__registry__:
            msg = f"Cannot create duplicate schema type {cls.regkey}."
            raise SchemaError(msg)
        cls.__registry__[cls.regkey] = cls
        super().__init__(name, bases, namespace, index=index)
        if payload is not None:
            if cls.payload_columns:
                msg = "A schema class can borrow payload columns or " + \
                      "define payload columns, but not both."
                raise SchemaError(msg)
            cls.__payload__ = payload
        else:
            cl_name = name + 'Payload'
            cl_namespace = cls.payload_columns

            def exec_body(ns):
                ns.update(cl_namespace)
                return ns

            cls.__payload__ = types.new_class(cl_name, (ColumnMap,),
                                              exec_body=exec_body)
        if cls.__payload__.payload_columns != cls.__payload__.__nxcolumns__:
            msg = "Ill-defined schema payload."
            raise SchemaError(msg)

    @property
    def regkey(cls):
        """Registration key."""
        return '.'.join([cls.__module__, cls.__name__])


@jsonio
class Schema(SchemaBase, metaclass=SchemaType):
    supertype = None

    def __init__(self, *columns, exclude=None):
        super().__init__()
        self.payload = self.__payload__(*columns, exclude=exclude)
        self._columns.update(self.payload._columns)

    @property
    def schema_columns(self):
        names = []
        for k, v in self.payload._columns.items():
            if v.generic:
                name = '#'.join((v.generic.name, k))
            else:
                name = k
            names.append(name)
        return names

    def to_dict(self, **kwargs):
        return {'class': type(self).regkey,
                'supertype': type(self).supertype,
                'columns': list(self.schema_columns)}

    @classmethod
    def from_dict(cls, dict_, **kwargs):
        try:
            regkey = dict_['class']
            columns = dict_['columns']
        except KeyError:
            msg = f"Invalid mapping."
            raise ValueError(msg)
        try:
            type_ = cls.__registry__[regkey]
        except KeyError:
            msg = f"Could not find a schema type {regkey}."
            raise TypeError(msg)
        return type_(*columns)


class IndexedColumnType(SchemaBaseType):

    def __new__(mcl, name, bases, namespace, index=None, column=None):
        return super().__new__(mcl, name, bases, namespace, index=index)

    def __init__(cls, name, bases, namespace, index=None, column=None):
        super().__init__(name, bases, namespace, index=index)

        if column is not None:
            if cls.payload_columns:
                msg = "An IndexedColumn type can either declare a column " + \
                      "or be supplied with it but not both."
                raise SchemaError(msg)
            cls.__column__ = column
        else:
            try:
                assert len(cls.payload_columns) == 1
            except AssertionError:
                if name == 'IndexedColumn':
                    return
                msg = "An IndexedColumn type features a single payload " + \
                      "column."
                raise SchemaError(msg)
            cls.__column__ = list(cls.payload_columns.values())[0]


class IndexedColumn(SchemaBase, metaclass=IndexedColumnType):
    """A mini-schema comprising an index and a single column, for series."""

    def __init__(self):
        super().__init__()
        self.column = self.__column__
        self._columns[self.column.name] = self.column


class MultiSchemaType(SchemaBaseType):

    def __new__(mcl, name, bases, namespace, index=None, families=None):
        return super().__new__(mcl, name, bases, namespace, index=index)

    def __init__(cls, name, bases, namespace, index=None, families=None):
        super().__init__(name, bases, namespace, index=index)

        if families is not None:
            if cls.column_families:
                msg = "A schema class can borrow families or define " + \
                      "column families, but not both."
                raise SchemaError(msg)
            cls.__families__ = families
            columns = index.__nxcolumns__.copy()
            columns.update(cls.__nxcolumns__)
            cls.__nxcolumns__ = columns
        else:
            fm_name = name + 'Families'
            fm_namespace = cls.column_families

            def exec_body(ns):
                ns.update(fm_namespace)
                return ns

            cls.__families__ = types.new_class(fm_name, (ColumnFamily,),
                                               exec_body=exec_body)
        if cls.__families__.column_families != cls.__families__.__nxcolumns__:
            msg = "Ill-defined multi-schema column families."
            raise SchemaError(msg)


class MultiSchema(SchemaBase, metaclass=MultiSchemaType):
    """An ordered mapping of names to column families.

    Params:
        *columns: column family names or column names declared in the type
            to include in the object. Column names in a particular family are
            specified in the form 'family.name'. The specification '*.name'
            is also acceptable if the name is common to all families, in
            which case it applies to all.
            If empty, then all families and columns are included. Passing
            unrecognized names will raise a SchemaError.
        exclude: conversely, it is possible to pass an iterable of excluded
            families and/or columns.
    Raises:
        SchemaError if unknown family or column names are used or if a
        required column is excluded.
    """

    @classmethod
    def _parse_colnames(cls, *names):
        """Utility function to parse column enumerations."""
        result = defaultdict(set)
        for string in names:
            try:
                fam, col = string.split(".")
            except ValueError:
                fam = string
                result[fam] = set()
            else:
                if fam == "*":
                    for f in cls.__nxcolumns__:
                        result[f].add(col)
                else:
                    result[fam].add(col)
        return result

    def __init__(self, *columns, exclude=None):
        super().__init__()
        families = self.__families__.__nxcolumns__
        required_families = {n: f for n, f in families.items() if f.required}
        if columns:
            include = self._parse_colnames(*columns)
        else:
            include = {f: set(v) for f, v in families.items()}
        if exclude is None:
            exclude = {}
        else:
            exclude = self._parse_colnames(*exclude)
        colnames = fun.merge_setmaps(include, exclude)
        if any(n not in families for n in colnames):
            msg = f"Unrecognized family names passed to {type(self)}."
            raise SchemaError(msg)
        if any(n not in colnames for n in required_families):
            msg = f"Cannot instantiate column map without required " + \
                  f"families {required_families}"
            raise SchemaError(msg)
        for fam, cols in colnames.items():
            family = families[fam]
            if any(c not in family for c in cols):
                msg = f"Unrecognized column names passed to {type(self)}."
                raise SchemaError(msg)

        for k, v in families.items():
            included = include.get(k, set())
            excluded = exclude.get(k, set())
            included -= excluded
            if not included:
                continue
            payload = type(v)(*included)
            self._columns[k] = payload
            setattr(self, k, payload)
        if not self._columns:
            msg = "Cannot instantiate column-less schema."
            raise SchemaError(msg)
        self.families = OrderedDict((k, v) for k, v in self._columns.items()
                                    if isinstance(v, ColumnFamily))

# =============================================================================
# Concrete Indexes and Schemas
# =============================================================================


class TimeSeriesIndex(SchemaIndex):
    id = String(index='nominal')
    datetime = DateTime(tz='UTC', index='sequential')

    def __rowkey__(self, index):
        id, dt = index
        postfix = str(int(1e6 * (nxtime.MAX_TIMESTAMP - dt.timestamp())))
        return '#'.join((id, postfix))

    def __rowidx__(self, key):
        id, postfix = key.split('#')
        dt = pd.Timestamp(1e6 * nxtime.MAX_TIMESTAMP - int(postfix),
                          tz='UTC', unit='us')
        return (id, dt)


class LogsSchema(Schema, index=TimeSeriesIndex):
    """Base schema for time series."""
    xindex = False  # xindex requires a previous record lookup in range queries
    twin_index = False  # twin_index has forward and backward indexes


class SampleLogsSchema(LogsSchema):
    """Schema for data samples accumulation."""
    supertype = 'sample'


class EventLogsSchema(LogsSchema):
    """Schema for events."""
    supertype = 'event'
    label = EventLabel()


class StateLogsSchema(LogsSchema):
    """Schema for state transitions."""
    supertype = 'state'
    xindex = True
    label = StateLabel()


class SessionLogsSchema(LogsSchema):
    """Schema for sessions -featuring a duration from the start datetime."""
    supertype = 'session'
    xindex = True
    duration = TimeDelta(required=True)
    label = SessionLabel()


class PeriodLogsSchema(LogsSchema):
    """Schema for periodic summaries."""
    supertype = 'period'
    xindex = True
    freq = None  # Frequency


class MultiTimeSeriesIndex(SchemaIndex):
    id = String(index='nominal')
    datetime = DateTime(tz='UTC', index='sequential')
    label = Categorical(index='suffix')

    def __rowkey__(self, index):
        id, dt, label = index
        postfix = str(int(1e6 * (nxtime.MAX_TIMESTAMP - dt.timestamp())))
        return '#'.join((id, postfix, label))

    def __rowidx__(self, key):
        id, postfix, label = key.split('#')
        dt = pd.Timestamp(1e6 * nxtime.MAX_TIMESTAMP - int(postfix),
                          tz='UTC', unit='us')
        return (id, dt, label)


TimeSeriesIndexes = (TimeSeriesIndex, MultiTimeSeriesIndex)


class MultiLogsSchema(LogsSchema, index=MultiTimeSeriesIndex):
    """Base schema for multi-time series."""
    pass


class MultiEventLogsSchema(MultiLogsSchema):
    supertype = 'event'


class MultiSessionLogsSchema(MultiLogsSchema):
    supertype = 'session'
    twin_index = True
