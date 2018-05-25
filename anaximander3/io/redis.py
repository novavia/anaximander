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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json

import attr
import pandas as pd
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

    @property
    def threadpoolsize(self):
        return (self.instance.connection_pool.max_connections // 3) + 1

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
        keys = list(self.instance.scan_iter(match=self.name + '#*'))
        if keys:
            self.instance.delete(*keys)

    def __query__(self, *columns, **kwargs):
        return RedisQuery(self, *columns, **kwargs)

    def __insert__(self, id, recs, pipe=None, **kwargs):
        storage_key = self.name + '#' + str(id)
        data = {json.dumps(r.payload): r.datetime.timestamp() for r in recs}
        client = pipe if pipe is not None else self.instance
        client.zadd(storage_key, **data)

    def __append__(self, logs, pipe=None, **kwargs):
        self.insert(*list(logs))

    # TODO: benchmark piping against threads
    def insert(self, *records, **kwargs):
        """Inserts one or more records into table."""
        rmap = defaultdict(list)
        for r in records:
            rmap[r.id].append(r)
        n = min(len(rmap), self.threadpoolsize)
        with ThreadPoolExecutor(n) as executor:
            executor.map(lambda i: self.__insert__(i[0], i[1], **kwargs),
                         rmap.items())

    # Note: in this form, this should significantly underperforms inserts
    def __update__(self, record, pipe=None, **kwargs):
        storage_key = self.name + '#' + str(record.id)
        score = record.datetime.timestamp()
        new_load = record.payload
        client = pipe if pipe is not None else self.instance

        try:
            elm = client.zrangebyscore(storage_key, score, score)[0]
        except IndexError:
            client.zadd(storage_key, score, json.dumps(new_load))
        else:
            client.zrem(storage_key, elm)
            load = json.loads(elm)
            load.update(new_load)
            client.zadd(storage_key, score, json.dumps(load))

    def __delete__(self, idx, pipe=None, **kwargs):
        id, datetime = idx
        storage_key = self.name + '#' + str(id)
        score = datetime.timestamp()
        client = pipe if pipe is not None else self.instance
        try:
            elm = client.zrangebyscore(storage_key, score, score)[0]
        except IndexError:
            pass
        else:
            client.zrem(storage_key, elm)

    # Note: try piping as an alternative to threads
    def delete(self, *idxs, **kwargs):
        """Deletes the specified rows based on their index."""
        n = self.threadpoolsize
        with ThreadPoolExecutor(n) as executor:
            executor.map(lambda i: self.__delete__(i, **kwargs), idxs)

    def __discard__(self, id, dt_range, pipe=None, **kwargs):
        """Deletes ranges of rows within dt_range for a given id."""
        storage_key = self.name + '#' + str(id)
        lower, upper = (t.timestamp() for t in dt_range)
        # Remove a nanosecond as a hack to exclude upper bound.
        upper -= 1e-9
        client = pipe if pipe is not None else self.instance
        client.zremrangebyscore(storage_key, lower, upper)

    def discard(self, *, id, datetime=(None, None), **kwargs):
        """Deletes ranges of rows within dt_range for specified ids."""
        q = self.query(id=id, datetime=datetime)
        if isinstance(q.id_range, rge.Level):
            self.__discard__(q.id_range.level, q.dt_range)
        else:
            n = min(len(q.id_range), self.threadpoolsize)
            with ThreadPoolExecutor(n) as executor:
                executor.map(lambda id: self.__discard__(id, q.dt_range,
                                                         **kwargs),
                             q.id_range)

    def __record__(self, idx, pipe=None, **kwargs):
        id, datetime = idx
        storage_key = self.name + '#' + str(id)
        score = datetime.timestamp()
        client = pipe if pipe is not None else self.instance
        data = client.zrangebyscore(storage_key, score, score)
        if not data:
            raise EmptyQueryException()
        return {'data': json.loads(data[0])}


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
        lower, upper = (t.timestamp() for t in self.dt_range)
        limit = int(fun.get(limit, 1e5))
        start, num = 0, limit
        client = pipe if pipe is not None else self.table.instance

        def sequence(id):
            storage_key = self.table.name + '#' + str(id)
            if self.table.reverse:
                results = client.zrevrangebyscore(storage_key,
                                                  upper, lower,
                                                  start, num,
                                                  withscores=True)
            else:
                results = client.zrangebyscore(storage_key,
                                               lower, upper,
                                               start, num,
                                               withscores=True)
            return results

        with ThreadPoolExecutor(len(id_range)) as executor:
            sequence_results = executor.map(sequence, id_range)

        for id, sr in zip(id_range, sequence_results):
            for data, score in sr:
                if score == upper:
                    continue
                idx = (id,
                       pd.Timestamp.utcfromtimestamp(score).tz_localize('utc'))
                yield idx, {'data': json.loads(data)}
