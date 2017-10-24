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
import datetime as dt

from google.cloud._helpers import _to_bytes
from google.cloud.bigtable._generated import (
    bigtable_pb2 as data_messages_v2_pb2)
from grpc._channel import _Rendezvous
from google.cloud.bigtable.client import Client
import google.cloud.bigtable.table as gc_big_table
from google.cloud.bigtable.column_family import MaxAgeGCRule
from google.cloud.bigtable.row_data import PartialRowsData
from google.cloud.bigtable.instance import Instance
from google.cloud.happybase.pool import ConnectionPool
from google.cloud.happybase.table import Table as HbTable
from google.cloud.bigtable.row_filters import ColumnQualifierRegexFilter, \
    RowFilterChain, RowFilterUnion

from ..utilities import functions as fun, nxattr, xprops
from ..utilities.nxtime import MAX_TIMESTAMP
from ..meta import prototype, metacharacter
from .schema import Schema
from .table import DataTable, DataQuery, DataQueryException, \
    DataTableWriteException, DataTableAdminException, EmptyQueryException

__all__ = ['BigTableDataTable', 'BigTableQuery', 'BigTableQueryException',
           'BigTableInsertException', 'Client', 'Instance']

# =============================================================================
# Monkeypatching of Table
# =============================================================================


def _create_row_request(table_name, row_key=None, start_key=None, end_key=None,
                        filter_=None, limit=None, reverse=False):
    """Creates a request to read rows in a table.

    :type table_name: str
    :param table_name: The name of the table to read from.

    :type row_key: bytes
    :param row_key: (Optional) The key of a specific row to read from.

    :type start_key: bytes
    :param start_key: (Optional) The beginning of a range of row keys to
                      read from. The range will include ``start_key``. If
                      left empty, will be interpreted as the empty string.

    :type end_key: bytes
    :param end_key: (Optional) The end of a range of row keys to read from.
                    The range will not include ``end_key``. If left empty,
                    will be interpreted as an infinite string.

    :type filter_: :class:`.RowFilter`
    :param filter_: (Optional) The filter to apply to the contents of the
                    specified row(s). If unset, reads the entire table.

    :type limit: int
    :param limit: (Optional) The read will terminate after committing to N
                  rows' worth of results. The default (zero) is to return
                  all results.

    :type reverse: bool
    :param reverse: if True, then the request is closed wrt. end key and
        open wrt. to the start key.

    :rtype: :class:`data_messages_v2_pb2.ReadRowsRequest`
    :returns: The ``ReadRowsRequest`` protobuf corresponding to the inputs.
    :raises: :class:`ValueError <exceptions.ValueError>` if both
             ``row_key`` and one of ``start_key`` and ``end_key`` are set
    """
    request_kwargs = {'table_name': table_name}
    if (row_key is not None and
            (start_key is not None or end_key is not None)):
        raise ValueError('Row key and row range cannot be '
                         'set simultaneously')
    range_kwargs = {}
    if start_key is not None or end_key is not None:
        if start_key is not None:
            if reverse is True:
                range_kwargs['start_key_open'] = _to_bytes(start_key)
            else:
                range_kwargs['start_key_closed'] = _to_bytes(start_key)
        if end_key is not None:
            if reverse is True:
                range_kwargs['end_key_closed'] = _to_bytes(end_key)
            else:
                range_kwargs['end_key_open'] = _to_bytes(end_key)
    if filter_ is not None:
        request_kwargs['filter'] = filter_.to_pb()
    if limit is not None:
        request_kwargs['rows_limit'] = limit

    message = data_messages_v2_pb2.ReadRowsRequest(**request_kwargs)

    if row_key is not None:
        message.rows.row_keys.append(_to_bytes(row_key))

    if range_kwargs:
        message.rows.row_ranges.add(**range_kwargs)

    return message


gc_big_table._create_row_request = _create_row_request


def read_rows(self, start_key=None, end_key=None, limit=None,
              filter_=None, reverse=False):
    """Read rows from this table.

    :type start_key: bytes
    :param start_key: (Optional) The beginning of a range of row keys to
                      read from. The range will include ``start_key``. If
                      left empty, will be interpreted as the empty string.

    :type end_key: bytes
    :param end_key: (Optional) The end of a range of row keys to read from.
                    The range will not include ``end_key``. If left empty,
                    will be interpreted as an infinite string.

    :type limit: int
    :param limit: (Optional) The read will terminate after committing to N
                  rows' worth of results. The default (zero) is to return
                  all results.

    :type filter_: :class:`.RowFilter`
    :param filter_: (Optional) The filter to apply to the contents of the
                    specified row(s). If unset, reads every column in
                    each row.

    :type reverse: bool
    :param reverse: if True, then the request is closed wrt. end key and
        open wrt. to the start key.

    :rtype: :class:`.PartialRowsData`
    :returns: A :class:`.PartialRowsData` convenience wrapper for consuming
              the streamed results.
    """
    request_pb = _create_row_request(
        self.name, start_key=start_key, end_key=end_key, filter_=filter_,
        limit=limit, reverse=reverse)
    client = self._instance._client
    response_iterator = client._data_stub.ReadRows(request_pb)
    # We expect an iterator of `data_messages_v2_pb2.ReadRowsResponse`
    return PartialRowsData(response_iterator)


gc_big_table.Table.read_rows = read_rows

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
                  'timestamp': lambda x: str(int(1e6 * (MAX_TIMESTAMP -
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


class BigTableAdminException(DataTableAdminException):
    pass


class BigTableInsertException(DataTableWriteException):
    pass


@prototype
@nxattr.s(hash=False, repr=False)
class BigTableDataTable(DataTable):
    schema = metacharacter(validate=fun.subcheck(Schema))
    instance = nxattr.ib(validator=nxattr.validators.instance_of(Instance))
    name = nxattr.ib(validator=nxattr.validators.instance_of(str))
    # Data retention time, in days, defaulting to None (indefinite retention)
    retention = nxattr.ib(None, repr=False)
    # An optional maxrate to limit query size automatically
    # This functionality requires a rowcount method to be implemented on
    # the table's Schema
    maxrate = nxattr.ib(None, repr=False)
    threadpoolsize = 250  # size of thread pool for inserts

    def __attrs_post_init__(self):
        tract = self.schema.tract
        if tract is not None:
            tract._bigtable = self

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

    @property
    def project(self):
        return self.client.project

    @property
    def gc_rule(self):
        """Garbage collection rule, implementing retention policy."""
        if self.retention is not None:
            return MaxAgeGCRule(dt.timedelta(self.retention))

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

    @xprops.cachedproperty
    def reverse(self):
        """If True, then keys are accumulated in reverse order.

        This influences how queries are processed.
        """
        try:
            assert self.schema.timestamp.key == 'timestamp'
        except (AttributeError, AssertionError):
            return False
        else:
            return True

    def __create__(self):
        rval = self.table.create()
        gc_rule = self.gc_rule
        for family in self.columns:
            column_family = self.table.column_family(family, gc_rule=gc_rule)
            column_family.create()
        return rval

    def __reset_retention__(self):
        if not self.exists:
            msg = "Cannot run function on non-existing table {0}."
            raise BigTableAdminException(msg.format(self))
        gc_rule = self.gc_rule
        for f in self.table.list_column_families().values():
            f.gc_rule = gc_rule
            f.update()

    def __remove__(self):
        return self.table.delete()

    def __query__(self, *fields, **kwargs):
        return BigTableQuery(self, *fields, **kwargs)

    def __insert__(self, record, **kwargs):
        rowkey = self.rowkey(*record.keys)
        row = self.table.row(rowkey)
        dump = record.dump()
        for family, columns in self.columns.items():
            for col in columns:
                value = str(dump.get(col, ''))
                row.set_cell(family,
                             col.encode('utf-8'),
                             value.encode('utf-8'))
        try:
            row.commit()
        # Single retry
        except _Rendezvous:
            self.__insert__(record, **kwargs)

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

    def __record__(self, *keys, **kwargs):
        """Returns a raw record from primary key, in the form of a tuple.

        If the primary key is not found, raises an EmptyQueryException.
        """
        rowkey = self.rowkey(*keys)
        row = self.table.read_row(rowkey.encode())
        if row is None:
            raise EmptyQueryException()
        attrs = ChainMap(*[{k.decode('utf-8'): v[0].value.decode('utf-8')
                            for k, v in row.cells[family].items()}
                           for family in self.columns])
        fields = self.schema.fields
        return tuple(attrs[k] if k in attrs
                     else fields[k].default for k in fields)

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

    def __repr__(self):
        string = 'BigTableTable[name={nm}](instance={ins})'
        return string.format(ins=self.instance.name, nm=self.name)


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

    def _make_rowkeypairs(self):
        """Returns an iterable of rowkey pairs to slice the table."""
        return (self._rowkeypair(*g, seqkeyrange=self.seqkeyrange)
                for g in self.nskeygroups)

    def read_row(self, row):
        attrs = ChainMap(*[{k.decode('utf-8'): v[0].value.decode('utf-8')
                            for k, v in row.cells[family].items()}
                           for family in self.columns])
        return tuple(attrs[k] if k in attrs else
                     self.table.schema.fields[k].default for k in self.fields)

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

    def first(self, **kwargs):
        kwargs['limit'] = 1
        return super().first(**kwargs)

    def __fetch__(self, maxrows=None, maxraise=False, limit=None):
        """Fetch primitive.

        Params:
            maxrows: limits the number of rows. If None and the table has
                a specified maxrate and sequential key, maxrows is computed
                automatically. An absolute limit of __maxrows__ is set by
                default.
            maxraise: raises if maxrows is exceeded.
            limit (int): limits the number of rows scanned by Bigtable on
                a per-non-sequential key basis. The default reads all rows.
        """
        rowkeypairs = self._make_rowkeypairs()
        row_groups = [self.table.table.read_rows(s, e,
                                                 filter_=self.rowfilter,
                                                 reverse=self.table.reverse,
                                                 limit=limit)
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
                for row in g.rows.values():
                    yield self.read_row(row)
                    row_count += 1
                    if row_count >= maxrows:
                        if maxraise:
                            msg = "Query exceeds maxrows {}".format(maxrows)
                            raise BigTableMaxRowsException(msg)
                        break
