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

from collections import defaultdict, OrderedDict, ChainMap
from concurrent.futures import ThreadPoolExecutor
from itertools import product

from grpc._channel import _Rendezvous
from google.cloud.bigtable.client import Client
from google.cloud.bigtable.instance import Instance
from google.cloud.happybase.pool import ConnectionPool
from google.cloud.happybase.table import Table as HbTable
from google.cloud.bigtable.row_filters import ColumnQualifierRegexFilter, \
    RowFilterChain, RowFilterUnion

from ..utilities import functions as fun, nxattr, xprops
from ..utilities.nxtime import MAX_TIMESTAMP
from ..meta import prototype, metacharacter
from .schema import Schema
from .annotations import interval
from .data import NxData
from .table import DataTable, DataQuery, DataQueryException, \
    DataTableWriteException

__all__ = ['BigTableDataTable', 'BigTableQuery', 'BigTableQueryException',
           'BigTableInsertException', 'Client', 'Instance']

# =============================================================================
# Schema specification and table creation
# =============================================================================


def btcolumns(*fields):
    """Returns columns from fields.

    This returns a dictionary of lists of column names, where the keys
    are the column family names. Column families can be specified in the
    metadata of schema fields.
    """
    columns = defaultdict(list)
    for field in fields:
        columns[field.family].append(field.name)
    return columns


def keymaker(schema):
    """Returns a make_key function from a schema.

    By convention, the rowkey assembles non-sequential keys in order
    first, then appends the sequential key if any. However, the expected
    sequence of *args in the rowkey function follows the same ordering
    as the key field declaration in the schema.
    """
    strategies = {'hash': lambda x: str(hash(x)),
                  'reverse': lambda x: str(x)[::-1],
                  'timestamp': lambda x: str(int(1e6 * (MAX_TIMESTAMP - \
                                                        x.timestamp()))),
                  'pmatsemit': lambda x: str(int(1e6 * x.timestamp()))[::-1]}
    keyfuncs = [strategies.get(f.key, lambda x: str(x))
                for f in schema.keys.values()]
    sqix = schema.seqkeyix
    keycount = len(schema.keys)

    def rowkeysequence(*keys):
        """Reorders key fields to match the rowkey sequence."""
        keys = iter(keys)
        for key in range(sqix):
            yield next(keys)
        sqkey = next(keys)
        for i in range(sqix + 1, keycount):
            yield next(keys)
        yield sqkey

    sequencer = (lambda *k: iter(k)) if sqix is None else rowkeysequence

    def rowkey(*keys):
        """Returns a row key from key field attributes.

        The expected order of *args is the same as the declaration
        sequence of key fields in the table's schema.
        """
        return '#'.join(sequencer(*[f(a) for f, a in zip(keyfuncs, keys)]))
   
    return rowkey

# =============================================================================
# Table implementation
# =============================================================================


class BigTableInsertException(DataTableWriteException):
    pass


@prototype
@nxattr.s(hash=False)
class BigTableDataTable(DataTable):
    schema = metacharacter(validate=fun.subcheck(Schema))
    instance = nxattr.ib(validator=nxattr.validators.instance_of(Instance))
    name = nxattr.ib(validator=nxattr.validators.instance_of(str))
    # An optional maxrate to limit query size automatically
    # This functionality requires a rowcount method to be implemented on
    # the table's Schema
    maxrate = nxattr.ib(None)
    threadpoolsize = 250  # size of thread pool for inserts

    @xprops.cachedproperty
    def table(self):
        """Caches an instance of table in the bigtable API."""
        table = self.instance.table(self.name)
        return table

    @property
    def exists(self):
        try:
            next(self.table.sample_row_keys())
        except StopIteration:
            return True
        except:
            return False
        else:
            return True

    @property
    def table_id(self):
        return self.table.table_id

    @property
    def client(self):
        return self.instance._client

    @xprops.cachedproperty
    def pool(self):
        """A connection pool from the happybase API."""
        return ConnectionPool(1, instance=self.instance)

    @xprops.cachedproperty
    def columns(self):
        """A dictionary of the form {family:[column names]}."""
        return btcolumns(*self.schema.fields.values())

    @xprops.cachedproperty
    def rowkey(self):
        """Function that turns a record's key attributes into a row key."""
        return keymaker(self.schema)        

    def __create__(self):
        rval = self.table.create()
        for family in self.columns:
            column_family = self.table.column_family(family)
            column_family.create()
        return rval

    def __remove__(self):
        return self.table.delete()

    def __query__(self, *fields, **kwargs):
        return BigTableQuery(self, *fields, **kwargs)

    def __insert__(self, record, **kwargs):
        rowkey = self.rowkey(*record.keys)
        row = self.table.row(rowkey)
        for family, columns in self.columns.items():
            for col in columns:
                value = getattr(record, col, '')
                if isinstance(value, NxData):
                    value = str(value.data)
                else:
                    value = str(value)
                row.set_cell(family,
                             col.encode('utf-8'),
                             value.encode('utf-8'))
        try:
            row.commit()
        except _Rendezvous:
            raise BigTableInsertException()

    def __append__(self, frame, **kwargs):
        records = frame.to_records()
        n = self.threadpoolsize
        with ThreadPoolExecutor(n) as executor:
            executor.map(self.__insert__, records)

    # Redefinition of DataTable.insert to take advantage of threads
    def insert(self, *records, **kwargs):
        """Inserts one or more records into table."""
        n = self.threadpoolsize
        with ThreadPoolExecutor(n) as executor:
            executor.map(self.__insert__, records)

    def _hbase_append__(self, frame, **kwargs):
        """Not in use because the connection pool is an illusion.
        
        In actuality this code makes individual inserts with the same
        row.commit programmed in __insert__. This is very slow because
        every commit is a blocking I/O operation.
        """
        keys = frame.data[list(self.schema.keynames)].values
        rowkeys = [self.rowkey(*k) for k in keys]
        rows = [tuple(str(v) for v in row) for row in frame.data.values]
        cols = tuple(':'.join((f, c)) for f, cols in self.columns.items()
                     for c in cols)
        data = (dict(zip(cols, (s.encode('utf-8') for s in r))) for r in rows)
        with self.pool.connection() as connection:
            table = HbTable(self.table_id, connection)
            batch = table.batch(transaction=True)
            for rowkey, rowdata in zip(rowkeys, data):
                batch.put(rowkey, rowdata)
            batch.send()

    def maxrows(self, start, end):
        """Estimates max. number of rows for single non-sequential key.

        This requires the schema to implement a rowcount function.
        """
        if self.maxrate is None:
            return NotImplemented
        return self.schema.rowcount(start, end, self.maxrate)


class BigTableQueryException(DataQueryException):
    pass


class BigTableMaxRowsException(BigTableQueryException):
    """Raised when queries reach specified maxrows."""
    pass


class BigTableQuery(DataQuery):
    __maxrows__ = 1e5  # Default limit for all queries
    
    def __init__(self, *fields, **quargs):
        super().__init__(*fields, **quargs)
        if not all(nskey in self.quargs for nskey in self.table.schema.nskeys):
            msg = "Target values for all non-sequential key fields must \
                   be specified in a BigTable Query."
            raise BigTableQueryException(msg)

    @xprops.cachedproperty
    def columns(self):
        """Returns self.fields organized by column family."""
        fields = (self.table.schema.fields[f] for f in self.fields)
        return btcolumns(*fields)      

    @xprops.cachedproperty
    def rowfilter(self):
        """Returns a row filter to be applied to query reads."""
        colfilters = [ColumnQualifierRegexFilter(f.encode('utf-8'))
                      for f in self.fields]
        colfilter = RowFilterUnion(colfilters)
        rgefields = self.table.schema.nonkeyfieldnames
        rangeargs = {k: v for k, v in self.quargs.items() if k in rgefields}
        rgefilters = [v.btfilter(k) for k, v in rangeargs.items()]
        # XXX: range filtering on non-key columns is disabled. Unsure how
        # to make this works since each range filter incorporates a
        # column filter... I.e. how do we apply a criteria on a particular
        # column while still returning other columns in the query results?
        rgefilters = []
        if rgefilters:
            return RowFilterChain(rgefilters + [colfilter])
        else:
            return colfilter

    def _rowkeypair(self, *nskeys, seqkeyrange=None):
        """Makes a pair of rowkeys to enable a table selection.

        Params:
            *nskeys: an iterable of non-sequential keys, in the same order
                as declared in the schema.
            sqk_interval: an optional interval (of type nxrange.Interval)
                that applies to the schema's sequential key, if any.
        Returns:
            a tuple containing a start and end rowkeys.
        """
        rowkey = self.table.rowkey  # rowkey function
        if seqkeyrange is None:
            return (rowkey(*nskeys),) * 2
        startkeys, endkeys = list(nskeys), list(nskeys)
        ix = self.table.schema.seqkeyix
        startkeys.insert(ix, seqkeyrange.lower)
        endkeys.insert(ix, seqkeyrange.upper)
        return tuple(sorted((rowkey(*startkeys), rowkey(*endkeys))))

    @xprops.cachedproperty
    def nskeygroups(self):
        """The combinations of non-sequential key groups."""
        nskeys = self.table.schema.nskeys
        nskeyquargs = OrderedDict(((k, self.quargs[k]) for k in nskeys))
        if nskeys:
            return list(product(*nskeyquargs.values()))
        else:
            return [tuple()]

    @xprops.cachedproperty
    def seqkeyrange(self):
        """The range for the sequential key, if applicable."""
        seqkey = self.table.schema.seqkey
        seqkeyfield = self.table.schema.keys.get(seqkey, None)        
        if seqkeyfield is not None:
            try:
                return self.quargs[seqkey]
            except KeyError:
                return interval(ref=seqkeyfield)
        else:
            return None

    def _make_rowkeypairs(self):
        """Returns an iterable of rowkey pairs to slice the table."""
        return (self._rowkeypair(*g, seqkeyrange=self.seqkeyrange)
                for g in self.nskeygroups)

    def read_row(self, row):
        attrs = ChainMap(*[{k.decode('utf-8'): v[0].value.decode('utf-8')
                            for k, v in row.cells[family].items()}
                           for family in self.columns])
        return tuple(attrs[k] for k in self.fields)

    @property
    def _maxrows(self):
        """A default value for maxrows."""
        if self.seqkeyrange is None:
            return self.__maxrows__
        max_per_group = self.table.maxrows(*self.seqkeyrange)
        if max_per_group is NotImplemented:
            return self.__maxrows__
        maxrows = max_per_group * len(self.nskeygroups)
        return min((maxrows, self.__maxrows__))        
    
    def __fetch__(self, maxrows=None, maxraise=False):
        """Fetch primitive.
        
        Params:
            maxrows: limits the number of rows. If None and the table has
                a specified maxrate and sequential key, maxrows is computed
                automatically. An absolute limit of __maxrows__ is set by
                default.
            maxraise: raises if maxrows is exceeded.
        """
        rowkeypairs = self._make_rowkeypairs()
        row_groups = [self.table.table.read_rows(s, e, filter_=self.rowfilter)
                      for s, e in rowkeypairs]
        maxrows = fun.get(maxrows, self._maxrows)
        row_count = 0
        for g in row_groups:
            while row_count < maxrows:
                g._rows = OrderedDict()
                try:
                    g.consume_next()
                except StopIteration:
                    break
                except _Rendezvous:
                    raise BigTableQueryException()
                for row in g.rows.values():
                    yield self.read_row(row)
                    row_count += 1
                    if row_count >= maxrows:
                        if maxraise:
                            msg = "Query exceeds maxrows {}".format(maxrows)
                            raise BigTableMaxRowsException(msg)
                        break
