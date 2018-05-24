#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides an interface from the data package to a redis server.

Here we wrap Redis capabilities in a rather liberal fashion by recreating
the concept of a table with keys and sequential data, which under the hood
implement Redis Sorted Sets.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from concurrent.futures import ThreadPoolExecutor
import json

import attr
from redis import StrictRedis, RedisError

from ..utilities import xprops, nxrange as rge, functions as fun
from ..data.nxschema import Schema, MultiSchema, TimeSeriesIndex
from .table import DataTable, DataQuery, DataQueryException, \
    DataTableWriteException, DataTableAdminException, EmptyQueryException

__all__ = ['RedisDataTable', 'RedisQuery', 'RedisQueryException',
           'RedisInsertException', 'client']


# =============================================================================
# Client factory
# =============================================================================


def client(host, port, password, max_connections, rolling=True):
    """Instantiates a Redis client.

    Params:
        rolling: if True, tables are versioned to enable rolling flushes.
            This should be set to False when Redis is used on a session-basis.
    """
    redis_client = StrictRedis(host=host, port=port, password=password,
                               max_connections=max_connections)
    redis_client.rolling = rolling
    return redis_client

# =============================================================================
# Table implementation
# =============================================================================


class RedisInsertException(DataTableWriteException):
    pass


@attr.s
class RedisDataTable(DataTable):
    instance = attr.ib(validator=attr.validators.instance_of(StrictRedis))
    name = attr.ib(validator=attr.validators.instance_of(str))
    schema = attr.ib(validator=attr.validators.instance_of((Schema,
                                                            MultiSchema)),
                     convert=lambda s: s() if isinstance(s, type) else s)
    # Data retention time, in days, defaulting to None (indefinite retention)
    retention = attr.ib(None, repr=False)
    threadpoolsize = 250  # size of thread pool for inserts

    @property
    def table_id(self):
        return self.name

    @property
    def client(self):
        return self.instance

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

    def storage_key(self, id):
        """Returns the storage key given an application id."""
        return '#'.join((self.name, id))

    def __exists__(self):
        """Tests existence of the resource. Must return True or False."""
        return True

    def __empty__(self):
        """Tests whether the resource is empty or not."""
        for k in self.instance.scan_iter(match=self.name + '#*'):
            try:
                next(self.instance.zscan_iter(k))
            except StopIteration:
                continue
            else:
                return False
        return True

    def __create__(self, **kwargs):
        pass

    def __drop__(self, force=False, **kwargs):
        pipe = self.instance.pipeline()
        keys = pipe.scan_iter(match=self.name + '#*')
        pipe.delete(*keys)
        pipe.execute()

    def __query__(self, *columns, pipe=None, **kwargs):
        return RedisQuery(self, *columns, pipe=pipe, **kwargs)

    def __insert__(self, record, pipe=None, retry=True, **kwargs):
        storage_key = self.name + '#' + str(record.id)
        score = record.datetime
        dump = record.json_dumps()
        if pipe is not None:
            return pipe.zadd(storage_key, score, dump)
        else:
            try:
                self.instance.zadd(storage_key, score, dump)
            except RedisError:
                if retry:
                    self.__insert__(record, retry=False, **kwargs)
                else:
                    raise

    def __append__(self, logs, pipe=None, **kwargs):
        records = list(logs)
        n = self.threadpoolsize
        with ThreadPoolExecutor(n) as executor:
            executor.map(self.__insert__, records, pipe=pipe, **kwargs)

    # Redefinition of DataTable.insert to take advantage of threads
    def insert(self, *records, **kwargs):
        """Inserts one or more records into table."""
        n = self.threadpoolsize
        with ThreadPoolExecutor(n) as executor:
            executor.map(self.__insert__, records, **kwargs)

    def __record__(self, idx, pipe=None, **kwargs):
        id, datetime = idx
        storage_key = self.name + '#' + str(id)
        if pipe is not None:
            data = pipe.zrangebyscore(storage_key, datetime, datetime)
        else:
            data = self.instance.zrangebyscore(storage_key, datetime, datetime)
        if not data:
            raise EmptyQueryException()
        return json.loads[data[0]]


class RedisQueryException(DataQueryException):
    pass


class RedisQuery(DataQuery):

    def __init__(self, *columns, **quargs):
        super().__init__(*columns, **quargs)
        if 'id' not in quargs:
            msg = "A Redis Query must specify an id range."
            raise RedisQueryException(msg)

    def first(self, **kwargs):
        kwargs['limit'] = 1
        return super().first(**kwargs)

    def __fetch__(self, pipe=None, limit=None, **kwargs):
        """Fetch primitive.

        Note that queries on sequential keys are closed on the left side
        and open on the right side.
        """
        if isinstance(self.id_range, rge.Level):
            id_range = [self.id_range.level]
        else:
            id_range = self.id_range
        lower, upper = self.dt_range
        limit = int(fun.get(limit, 1e5))
        start, num = 0, limit

        def sequence(id):
            storage_key = self.table.name + '#' + str(id)
            client = pipe if pipe is not None else self.table.instance
            if self.table.reverse:
                results = client.zrevrangebyscore(storage_key,
                                                  lower, upper,
                                                  start, num,
                                                  withscores=True)
            else:
                results = client.zrangebyscore(storage_key,
                                               lower, upper,
                                               start, num,
                                               withscores=True)
            return results

        with ThreadPoolExecutor(len(id_range)) as executor:
            sequence_results = executor.map(id_range)

        for sr in sequence_results:
            for data, score in sr:
                if score == upper:
                    continue
                yield json.loads(data)
