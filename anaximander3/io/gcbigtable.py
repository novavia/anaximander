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

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import datetime as dt

import attr
from google.cloud._helpers import _to_bytes
from google.cloud.bigtable._generated import (
    bigtable_pb2 as data_messages_v2_pb2)
from grpc._channel import _Rendezvous
from google.cloud.bigtable.client import Client
import google.cloud.bigtable.table as gc_big_table
from google.cloud.bigtable.column_family import MaxAgeGCRule
from google.cloud.bigtable.row_data import PartialRowsData
from google.cloud.bigtable.instance import Instance
from google.cloud.bigtable.row_filters import ColumnQualifierRegexFilter, \
    RowFilterUnion

from ..utilities import xprops, nxrange as rge, functions as fun
from ..data.nxschema import Schema, MultiSchema, TimeSeriesIndex
from ..data import records as rcd
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
                range_kwargs['end_key_closed'] = _to_bytes(start_key)
            else:
                range_kwargs['start_key_closed'] = _to_bytes(start_key)
        if end_key is not None:
            if reverse is True:
                range_kwargs['start_key_open'] = _to_bytes(end_key)
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
# Table implementation
# =============================================================================


class BigTableAdminException(DataTableAdminException):
    pass


class BigTableInsertException(DataTableWriteException):
    pass


@attr.s
class BigTableDataTable(DataTable):
    instance = attr.ib(validator=attr.validators.instance_of(Instance))
    name = attr.ib(validator=attr.validators.instance_of(str))
    schema = attr.ib(validator=attr.validators.instance_of((Schema,
                                                            MultiSchema)),
                     convert=lambda s: s() if isinstance(s, type) else s)
    # Data retention time, in days, defaulting to None (indefinite retention)
    retention = attr.ib(None, repr=False)
    threadpoolsize = 250  # size of thread pool for inserts

    @xprops.cachedproperty
    def table(self):
        """Caches an instance of table in the bigtable API."""
        table = self.instance.table(self.name)
        return table

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
    def reverse(self):
        """If True, then keys are accumulated in reverse order.

        This influences how queries are processed.
        """
        try:
            assert isinstance(self.schema.index, TimeSeriesIndex)
        except (AttributeError, AssertionError):
            return False
        else:
            return True

    def __exists__(self):
        """Tests existence of the resource. Must return True or False."""
        return self.name in [t.table_id for t in self.instance.list_tables()]

    def __empty__(self):
        """Tests whether the resource is empty or not."""
        try:
            next(self.table.sample_row_keys())
        except StopIteration:
            return True
        except:
            return False
        else:
            return True

    def __create__(self, **kwargs):
        rval = self.table.create()
        gc_rule = self.gc_rule
        if isinstance(self.schema, Schema):
            data = self.table.column_family('data', gc_rule=gc_rule)
            data.create()
        elif isinstance(self.schema, MultiSchema):
            for f in self.schema.families:
                family = self.table.column_family(f, gc_rule=gc_rule)
                family.create()
        return rval

    def __drop__(self, force=False, **kwargs):
        return self.table.delete()

    def __reset_retention__(self):
        if not self.exists:
            msg = "Cannot run function on non-existing table {0}."
            raise BigTableAdminException(msg.format(self))
        gc_rule = self.gc_rule
        for f in self.table.list_column_families().values():
            f.gc_rule = gc_rule
            f.update()

    def __query__(self, *columns, **kwargs):
        return BigTableQuery(self, *columns, **kwargs)

    def __insert__(self, record, retry=True, **kwargs):
        if not isinstance(record, rcd.Record):
            msg = f"Table insert requires a record, not a " + \
                  f"{type(record)} instance."
            raise TypeError(msg)
        if not isinstance(record.schema, type(self.schema)):
            msg = f"Incorrect record schema supplied to {self}."
            raise ValueError(msg)
        rowkey = record.key
        row = self.table.row(rowkey)
        dump = record.to_dict()
        for family in self.table.list_column_families():
            data = dump.get(family, {})
            for k, v in data.items():
                row.set_cell(family, k.encode('utf-8'), str(v).encode('utf-8'))
        try:
            row.commit()
        # Single retry
        except _Rendezvous:
            if retry:
                self.__insert__(record, retry=False, **kwargs)
            else:
                raise

    def __append__(self, logs, **kwargs):
        records = list(logs)
        n = min(len(records), self.threadpoolsize)
        with ThreadPoolExecutor(n) as executor:
            executor.map(lambda r: self.__insert__(r, **kwargs), records)

    # Redefinition of DataTable.insert to take advantage of threads
    def insert(self, *records, **kwargs):
        """Inserts one or more records into table."""
        n = min(len(records), self.threadpoolsize)
        with ThreadPoolExecutor(n) as executor:
            executor.map(lambda r: self.__insert__(r, **kwargs), records)

    def update(self, *records, **kwargs):
        """Updates rows in place from records, or inserts if not found."""
        return self.insert(*records, **kwargs)

    def __delete__(self, idx, retry=True, **kwargs):
        rowkey = self.schema.rowkey(idx)
        row = self.table.row(rowkey.encode())
        row.delete()
        try:
            row.commit()
        # Single retry
        except _Rendezvous:
            if retry:
                self.__delete__(idx, retry=False, **kwargs)
            else:
                raise

    def delete(self, *idxs, **kwargs):
        """Deletes the specified rows based on their index."""
        n = min(len(idxs), self.threadpoolsize)
        with ThreadPoolExecutor(n) as executor:
            executor.map(lambda i: self.__delete__(i, **kwargs), idxs)

    def __discard__(self, id, datetime, **kwargs):
        """Deletes ranges of rows within dt_range for specified ids."""
        query = self.query(id=id, datetime=datetime)
        rowkeypairs = list(query.rowkeypairs())

        def row_group(keys):
            start_key, end_key = keys
            return self.table.read_rows(start_key,
                                        end_key,
                                        reverse=self.reverse)

        n = min(len(rowkeypairs), self.threadpoolsize)
        with ThreadPoolExecutor(n) as executor:
            row_groups = executor.map(row_group, rowkeypairs)

        def delete_row(key, retry=True):
            row = self.table.row(key)
            row.delete()
            try:
                row.commit()
            except _Rendezvous:
                if retry:
                    delete_row(key, False)
                else:
                    raise

        for g in row_groups:
            while True:
                g._rows = OrderedDict()
                try:
                    g.consume_next()
                except StopIteration:
                    break
                n = min(len(g.rows), self.threadpoolsize)
                with ThreadPoolExecutor(n) as executor:
                    executor.map(delete_row, g.rows)

    def __record__(self, idx, **kwargs):
        rowkey = self.schema.rowkey(idx)
        row = self.table.read_row(rowkey.encode())
        if row is None:
            raise EmptyQueryException()
        data = {f: {k.decode(): v[0].value.decode() for k, v in d.items()}
                for f, d in row.cells.items()}
        return data


class BigTableQueryException(DataQueryException):
    pass


class BigTableQuery(DataQuery):

    def __init__(self, *columns, **quargs):
        super().__init__(*columns, **quargs)
        if 'id' not in quargs:
            msg = "A BigTable Query must specify an id range."
            raise BigTableQueryException(msg)

    @xprops.cachedproperty
    def rowfilter(self):
        """Returns a row filter to be applied to query reads."""
        colfilters = [ColumnQualifierRegexFilter(c.encode('utf-8'))
                      for c in self.schema]
        return RowFilterUnion(colfilters)

    def rowkeypairs(self):
        """A generator of rowkey pairs to slice the table."""
        if isinstance(self.id_range, rge.Level):
            id_range = [self.id_range.level]
        else:
            id_range = self.id_range
        lower, upper = self.dt_range
        for id_ in id_range:
            start_key = self.schema.rowkey((id_, lower))
            end_key = self.schema.rowkey((id_, upper))
            yield (start_key, end_key)

    def read_row(self, row):
        data = {f: {k.decode(): v[0].value.decode() for k, v in d.items()}
                for f, d in row.cells.items()}
        idx = self.schema.rowidx(row.row_key.decode())
        return (idx, data)

    def first(self, **kwargs):
        kwargs['limit'] = 1
        return super().first(**kwargs)

    def __fetch__(self, limit=None, **kwargs):
        """Fetch primitive.

        Params:
            limit (int): limits the number of rows per id scanned by Bigtable.
        """
        rowkeypairs = list(self.rowkeypairs())
        limit = int(fun.get(limit, 1e5))

        def row_group(keys):
            start_key, end_key = keys
            return self.table.table.read_rows(start_key,
                                              end_key,
                                              filter_=self.rowfilter,
                                              reverse=self.table.reverse,
                                              limit=limit)

        n = min(len(rowkeypairs), self.table.threadpoolsize)
        with ThreadPoolExecutor(n) as executor:
            row_groups = executor.map(row_group, rowkeypairs)

        for g in row_groups:
            while True:
                g._rows = OrderedDict()
                try:
                    g.consume_next()
                except StopIteration:
                    break
                for row in g.rows.values():
                    yield self.read_row(row)
