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
from google.cloud.bigtable.row_filters import ColumnQualifierRegexFilter, \
    RowFilterUnion

from ..utilities import xprops, nxrange as rge, functions as fun, nxtime
from ..data.nxschema import Schema, MultiSchema, TimeSeriesIndex
from .store import Store, DataTract, Query, QueryException, \
    WriteException, StorageAdminException, EmptyQueryException

__all__ = ['BigTableStore', 'BigDataTract', 'BigTableQuery',
           'BigTableQueryException', 'BigTableInsertException', 'Client',
           'Instance']

# =============================================================================
# Monkeypatching of Table
# =============================================================================


def _create_row_request(table_name, row_key=None, start_key=None, end_key=None,
                        filter_=None, limit=None, reverse=False, closed=False):
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

    :type closed: bool
    :param closed: if True, the request is closed on both ends. Otherwise
        open on end key, or on start key if reverse is True.

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
            if closed:
                if reverse is True:
                    range_kwargs['start_key_closed'] = _to_bytes(end_key)
                else:
                    range_kwargs['end_key_closed'] = _to_bytes(end_key)
            else:
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
              filter_=None, reverse=False, closed=False):
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
        open / closed wrt. to the start key depending on 'closed'.

    :type closed: bool
    :param closed: if True, the request is closed on both ends. Otherwise
        open on end key, or on start key if reverse is True.

    :rtype: :class:`.PartialRowsData`
    :returns: A :class:`.PartialRowsData` convenience wrapper for consuming
              the streamed results.
    """
    request_pb = _create_row_request(
        self.name, start_key=start_key, end_key=end_key, filter_=filter_,
        limit=limit, reverse=reverse, closed=closed)
    client = self._instance._client
    response_iterator = client._data_stub.ReadRows(request_pb)
    # We expect an iterator of `data_messages_v2_pb2.ReadRowsResponse`
    return PartialRowsData(response_iterator)


gc_big_table.Table.read_rows = read_rows

# =============================================================================
# Table implementation
# =============================================================================


class BigTableAdminException(StorageAdminException):
    pass


class BigTableInsertException(WriteException):
    pass


class BigTableStore(Store):

    def __interface__(self, instance_id=None, project_id=None, admin=False):
        self.client = Client(project=project_id, admin=admin)
        return self.client.instance(instance_id)

    def create(self, location, display_name=None, serve_nodes=None):
        if self.io.instance_id in [i.instance_id
                                   for i in self.client.list_instances()[0]]:
            msg = f"Instance {self.io.instance_id} already exists."
            raise BigTableAdminException(msg)
        self.io._cluster_location_id = location
        if display_name:
            self.io.display_name = display_name
        if serve_nodes:
            self._cluster_serve_nodes = serve_nodes
        self.io.create()

    def __repr__(self):
        p, i = self.client.project, self.io.instance_id
        return f'<BigTableStore project:{p} instance:{i}>'


class BigDataTract(DataTract):

    def __init__(self, store, title, retention=None, register=True):
        """Data retention time, in days, defaulting to None (indefinite)."""
        super().__init__(store, title, register)
        self._retention = dt.timedelta(retention) if retention else retention

    @xprops.cachedproperty
    def table(self):
        """Caches an instance of table in the bigtable API."""
        table = self.store.io.table(self.name)
        return table

    @property
    def table_id(self):
        return self.table.table_id

    @property
    def client(self):
        return self.store.io._client

    @property
    def project(self):
        return self.client.project

    @property
    def retention(self):
        return self._retention

    @retention.setter
    def retention(self, retention):
        self._retention = dt.timedelta(retention) if retention else retention
        if self.exists:
            self.__reset_retention__()

    @property
    def gc_rule(self):
        """Garbage collection rule, implementing retention policy."""
        if self.retention is not None:
            return MaxAgeGCRule(dt.timedelta(self.retention))

    def __exists__(self):
        """Tests existence of the resource. Must return True or False."""
        return self.name in [t.table_id for t in self.store.io.list_tables()]

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

    def __insert__(self, record, **kwargs):
        rowkey = record.key
        row = self.table.row(rowkey)
        dump = record.to_dict()
        for family in self.table.list_column_families():
            data = dump.get(family, {})
            for k, v in data.items():
                row.set_cell(family, k.encode('utf-8'), str(v).encode('utf-8'))
        return row

    def __append__(self, logs, **kwargs):
        records = list(logs)
        rows = [self.__insert__(r, **kwargs) for r in records]
        return self.table.mutate_rows(rows)

    def insert(self, *records, **kwargs):
        """Inserts one or more records into table."""
        rows = super().insert(*records, **kwargs)
        return self.table.mutate_rows(rows)

    __update__ = __insert__

    def update(self, *records, **kwargs):
        """Updates rows in place from records, or inserts if not found."""
        rows = super().update(*records, **kwargs)
        return self.table.mutate_rows(rows)

    def __delete__(self, idx, **kwargs):
        rowkey = self.schema.rowkey(idx)
        row = self.table.row(rowkey.encode())
        row.delete()
        return row

    def delete(self, *idxs, **kwargs):
        """Deletes the specified rows based on their index."""
        rows = super().delete(*idxs, **kwargs)
        return self.table.mutate_rows(rows)

    def __discard__(self, id, dt_range, **kwargs):
        """Deletes ranges of rows within dt_range for specified id."""
        start_key = self.schema.rowkey((id, dt_range.lower))
        end_key = self.schema.rowkey((id, dt_range.upper))
        try:
            group = self.table.read_rows(start_key, end_key, reverse=True)
        except _Rendezvous:
            group = self.table.read_rows(start_key, end_key, reverse=True)
        while True:
            group._rows = OrderedDict()
            try:
                group.consume_next()
            except StopIteration:
                break
            rows = [self.table.row(k) for k in group.rows]
            for r in rows:
                r.delete()
            self.table.mutate_rows(rows)

    def __record__(self, idx, **kwargs):
        rowkey = self.schema.rowkey(idx)
        try:
            row = self.table.read_row(rowkey.encode())
        # Single retry fixes most problems
        except _Rendezvous:
            row = self.table.read_row(rowkey.encode())
        if row is None:
            raise EmptyQueryException()
        data = {f: {k.decode(): v[0].value.decode() for k, v in d.items()}
                for f, d in row.cells.items()}
        return data


class BigTableQueryException(QueryException):
    pass


class BigTableQuery(Query):

    def __init__(self, tract, *columns, **quargs):
        super().__init__(tract, *columns, **quargs)
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
        """A generator of rowkey pairs to slice the tract."""
        if isinstance(self.id_range, rge.Level):
            id_range = [self.id_range.level]
        else:
            id_range = self.id_range
        lower, upper = self.dt_range
        bottom = nxtime.MIN
        for id_ in id_range:
            start_key = self.schema.rowkey((id_, lower))
            end_key = self.schema.rowkey((id_, upper))
            bottom_key = self.schema.rowkey((id_, bottom))
            yield (start_key, end_key, bottom_key)

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

        def rows(keys):
            start_key, end_key, bottom_key = keys
            try:
                group = self.tract.table.read_rows(start_key,
                                                   end_key,
                                                   filter_=self.rowfilter,
                                                   reverse=True,
                                                   limit=limit)
            except _Rendezvous:
                group = self.tract.table.read_rows(start_key,
                                                   end_key,
                                                   filter_=self.rowfilter,
                                                   reverse=True,
                                                   limit=limit)
            if self.schema.xindex:
                next_ = self.tract.table.read_rows(bottom_key,
                                                   start_key,
                                                   filter_=self.rowfilter,
                                                   reverse=True,
                                                   closed=True,
                                                   limit=1)
            else:
                next_ = None
            return group, next_

        for k in rowkeypairs:
            group, next_ = rows(k)
            key = None
            while True:
                group._rows = OrderedDict()
                try:
                    group.consume_next()
                except StopIteration:
                    break
                for key, row in group.rows.items():
                    yield self.read_row(row)
            if next_ is not None:
                try:
                    next_.consume_next()
                except StopIteration:
                    continue
                nx_key, nx_row = next(iter(next_.rows.items()))
                if not nx_key == key:
                    yield self.read_row(nx_row)
