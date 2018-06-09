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
import json

import pandas as pd
from redis import StrictRedis

from ..utilities import xprops, nxtime, nxrange as rge, functions as fun
from ..data import records as rcd, datalogs as dtl
from ..data.nxschema import TimeSeriesIndex
from .store import Store, Title, Tract, DataTract, Query, QueryException, \
    WriteException

__all__ = ['RedisStore', 'RedisTract', 'RedisDataQuery', 'RedisQueryException',
           'RedisInsertException']

NS = pd.Timedelta(nanoseconds=1)

# =============================================================================
# Tract implementation
# =============================================================================


class RedisInsertException(WriteException):
    pass


class RedisStore(Store):

    def __interface__(self, host, port, password):
        return StrictRedis(host=host, port=port, password=password)

    def __repr__(self):
        conn_kwargs = self.io.connection_pool.connection_kwargs
        h, p, db = conn_kwargs['host'], conn_kwargs['port'], conn_kwargs['db']
        return f'<RedisStore host:{h} port:{p} db:{db}>'


class RedisTract(Tract):

    @property
    def table_id(self):
        return self.name

    @property
    def client(self):
        return self.store.io

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
        return True

    def __empty__(self):
        """Tests whether the resource is empty or not."""
        for k in self.client.scan_iter(match=self.name + '#*'):
            try:
                next(self.client.zscan_iter(k))
            except StopIteration:
                continue
            else:
                return False
        return True

    def __create__(self, **kwargs):
        pass

    def __drop__(self, force=False, **kwargs):
        keys = list(self.client.scan_iter(match=self.name + '#*'))
        if keys:
            self.client.delete(*keys)


class RedisDataTract(DataTract, RedisTract):

    def __init__(self, store, title, max_range=None, max_count=None,
                 register=True):
        super().__init__(store, title, register)
        # Optional time depth, in minutes. Following an insertion, elements
        # that are older than the most recent insert minus the max_range are
        # removed. Note that this doesn't guarantee that the time range of
        # the elements in a sequence is less than max_range. In particular,
        # one could accumulate arbitrary old elements without limits. However
        # this is designed for buffering data with generally monotonously
        # increasing time stamps.
        self.max_range = pd.Timedelta(minutes=max_range) if max_range else None
        # Optional maximum element count. Oldest elements by score are removed
        # when the count is reached.
        self.max_count = max_count

    def storage_key(self, id):
        """Returns the storage key given an application id."""
        return '#'.join((self.name, id))

    def __query__(self, *columns, **kwargs):
        return RedisDataQuery(self, *columns, **kwargs)

    def __insert__(self, pipe, id, recs, **kwargs):
        if not recs:
            return
        storage_key = self.name + '#' + str(id)
        data = {json.dumps(r.payload): r.datetime.timestamp() for r in recs}
        pipe.zadd(storage_key, **data)
        if self.max_count:
            pipe.zremrangebyrank(storage_key, 0, -(self.max_count + 1))
        if self.max_range:
            max_ts = max(data.values())
            min_score = max_ts - self.max_range.total_seconds() + 1e-9
            pipe.zremrangebyscore(storage_key, '-inf', min_score)

    def insert(self, *records, validate=True, **kwargs):
        """Inserts one or more records into tract."""
        rmap = defaultdict(list)
        if validate:
            for record in records:
                if not isinstance(record, rcd.Record):
                    msg = f"Tract insert requires a record, not a " + \
                          f"{type(record)} instance."
                    raise TypeError(msg)
                if not isinstance(record.schema, type(self.schema)):
                    msg = f"Incorrect record schema supplied to {self}."
                    raise ValueError(msg)
                rmap[record.id].append(record)
        else:
            for record in records:
                rmap[record.id].append(record)

        pipe = self.client.pipeline()
        for id, recs in rmap.items():
            self.__insert__(pipe, id, recs, **kwargs)
        return pipe.execute()

    def __append__(self, pipe, logs, **kwargs):
        return self.insert(*list(logs), validate=False)

    def append(self, logs, **kwargs):
        pipe = self.client.pipeline()
        kwargs.setdefault('pipe', pipe)
        super().append(logs, **kwargs)

    def __update__(self, pipe, id, recs, **kwargs):
        storage_key = self.name + '#' + str(id)
        payloads = {r.datetime.timestamp(): r.payload for r in recs}
        min_score, max_score = min(payloads), max(payloads)
        extant = self.client.zrangebyscore(storage_key, min_score, max_score,
                                           withscores=True)
        extant = {s: data for data, s in extant}
        for score, payload in payloads.items():
            try:
                elm = extant[score]
            except KeyError:
                pipe.zadd(storage_key, score, json.dumps(payload))
            else:
                load = json.loads(elm)
                pipe.zrem(storage_key, elm)
                load.update(payload)
                pipe.zadd(storage_key, score, json.dumps(load))

        if self.max_count:
            pipe.zremrangebyrank(storage_key, 0, -(self.max_count + 1))
        if self.max_range:
            min_score = max_score - self.max_range.total_seconds() + 1e-9
            pipe.zremrangebyscore(storage_key, '-inf', min_score)

    def update(self, *records, **kwargs):
        """Updates rows in place from records, or inserts if not found."""
        rmap = defaultdict(list)
        for record in records:
            if not isinstance(record, rcd.Record):
                msg = f"Tract update requires a record, not a " + \
                      f"{type(record)} instance."
                raise TypeError(msg)
            if not isinstance(record.schema, type(self.schema)):
                msg = f"Incorrect record schema supplied to {self}."
                raise ValueError(msg)
            rmap[record.id].append(record)

        pipe = self.client.pipeline()
        for id, recs in rmap.items():
            self.__update__(pipe, id, recs, **kwargs)
        return pipe.execute()

    def __delete__(self, pipe, id, datetimes, **kwargs):
        storage_key = self.name + '#' + str(id)
        scores = [dt.timestamp() for dt in datetimes]
        min_score, max_score = min(scores), max(scores)
        records = self.client.zrangebyscore(storage_key,
                                            min_score, max_score,
                                            withscores=True)
        records = {s: data for data, s in records}
        for s in scores:
            try:
                pipe.zrem(storage_key, records[s])
            except KeyError:
                pass

    def delete(self, *idxs, **kwargs):
        """Deletes the specified rows based on their index."""
        rmap = defaultdict(list)
        for idx in idxs:
            rmap[idx[0]].append(idx[1])

        pipe = self.client.pipeline()
        for id, recs in rmap.items():
            self.__delete__(pipe, id, recs, **kwargs)
        return pipe.execute()

    def __discard__(self, pipe, id, dt_range, **kwargs):
        """Deletes ranges of rows within dt_range for a given id."""
        storage_key = self.name + '#' + str(id)
        lower, upper = (t.timestamp() for t in dt_range)
        # Remove a nanosecond as a hack to exclude upper bound.
        upper -= 1e-9
        pipe.zremrangebyscore(storage_key, lower, upper)

    def discard(self, *ids, datetime=(None, None), **kwargs):
        """Deletes ranges of rows within dt_range for specified ids."""
        dt_range = rge.time_range(datetime)
        pipe = self.client.pipeline()
        for id in ids:
            self.__discard__(pipe, id, dt_range, **kwargs)
        return pipe.execute()

    def __record__(self, pipe, idx, **kwargs):
        id, datetime = idx
        storage_key = self.name + '#' + str(id)
        score = datetime.timestamp()
        pipe.zrangebyscore(storage_key, score, score)

    def _record(self, idx, result):
        """Primitive for record and records."""
        if not result:
            msg = f"No row found with index {idx}."
            raise KeyError(msg)
        data = {'data': json.loads(result[0])}
        data['index'] = idx
        data['schema'] = self.schema
        return rcd.Record.from_dict(data)

    def record(self, *idx, **kwargs):
        """Returns a record from its index."""
        if len(idx) == 1:
            idx = idx[0]
        pipe = self.client.pipeline()
        self.__record__(pipe, idx, **kwargs)
        result = pipe.execute()[0]
        return self._record(idx, result)

    def records(self, *idxs, **kwargs):
        """Returns a records iterator from their index."""
        pipe = self.client.pipeline()
        for idx in idxs:
            self.__record__(pipe, idx, **kwargs)
        results = pipe.execute()
        for idx, result in zip(idxs, results):
            yield self._record(idx, result)


class RedisQueryException(QueryException):
    pass


class RedisDataQuery(Query):

    def __init__(self, tract, *columns, **quargs):
        super().__init__(tract, *columns, **quargs)
        if 'id' not in quargs:
            msg = "A Redis Query must specify an id range."
            raise RedisQueryException(msg)

    def first(self, **kwargs):
        kwargs['limit'] = 1
        return super().first(**kwargs)

    def __fetch__(self, limit=None, **kwargs):
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
        pipe = self.tract.client.pipeline()

        if self.tract.reverse:
            for id in id_range:
                storage_key = self.tract.storage_key(id)
                pipe.zrevrangebyscore(storage_key, upper, lower,
                                      start, num, withscores=True)
        else:
            for id in id_range:
                storage_key = self.tract.storage_key(id)
                pipe.zrangebyscore(storage_key, lower, upper,
                                   start, num, withscores=True)

        sequence_results = pipe.execute()
        for id, sr in zip(id_range, sequence_results):
            for data, score in sr:
                if score == upper:
                    continue
                idx = (id,
                       pd.Timestamp.utcfromtimestamp(score).tz_localize('utc'))
                yield idx, {'data': json.loads(data)}


class RedisBuffer(RedisTract):
    """A specialized storage structure that combines data & metadata.

    The metadata is stored as a hash with the following structures:
        lower: timestamp  # lower bound of the buffer
        upper: timestamp  # upper bound of the buffer
        certification: timestamp  # certification line within buffer
        [subscriber title]: timestamp  # for each subscribing buffer, their
            own certification line, which determines data eviction.
    """

    def __init__(self, store, title, max_range=None, register=True):
        super().__init__(store, title, register)
        self.max_range = pd.Timedelta(minutes=max_range) if max_range else None
        data_title = Title(title.name + '_data', title.schema)
        self.data_tract = RedisDataTract(store, data_title, max_range,
                                         register=False)

    def __get_data__(self, pipe, id, lower=None):
        dt_range = rge.time_range((lower, None))
        lower, upper = (t.timestamp() for t in dt_range)
        if self.tract.reverse:
            storage_key = self.tract.storage_key(id)
            pipe.zrevrangebyscore(storage_key, upper, lower, withscores=True)
        else:
            storage_key = self.tract.storage_key(id)
            pipe.zrangebyscore(storage_key, lower, upper, withscores=True)

    def sequence(self, id, lower=None):
        """Fetches data for id, optionally above lower datetime."""
        pipe = self.client.pipeline()
        self.__get_metadata__(pipe, id)
        self.__get_data__(pipe, id, lower=lower)
        meta_, data_ = pipe.execute()
        metadata = {k: pd.to_datetime(v, utc=True) for k, v in meta_.items()}
        dt_range = rge.time_range(metadata.pop('lower', None),
                                  metadata.pop('upper', None))
        consumption = nxtime.MAX
        for k in metadata.keys():
            if k != 'certification':
                v = metadata.pop(k)
                consumption = min((consumption, v))
        metadata['dt_range'] = dt_range
        metadata['consumption'] = consumption
        datetime, payload = [], []
        for d, score in data_:
            dt = pd.Timestamp.utcfromtimestamp(score).tz_localize('utc')
            datetime.append(dt)
            payload.append(json.load(d))
        data = pd.DataFrame(payload)
        data['id'] = id
        data['datetime'] = datetime
        return dtl.Sequence(data, schema=self.schema, id_range=id, **metadata)

    def meta_storage_key(self, id):
        """Returns the storage key for metadata."""
        return '#'.join((self.name + '_meta', id))

    def __get_metadata__(self, pipe, id):
        pipe.hgetall(self.meta_storage_key(id))

    def metadata(self, id):
        """Queries and returns the metadata for supplied id."""
        pipe = self.client.pipeline()
        self.__get_metadata__(pipe, id)
        result = pipe.execute()[0]
        return {k: pd.to_datetime(v, utc=True) for k, v in result.items()}

    def __update_range__(self, pipe, id, dt_range, certificate=None):
        lower, upper = (t.timestamp() for t in dt_range)
        mapping = {'lower': lower, 'upper': upper}
        if certificate is not None:
            mapping['certification'] = certificate.timestamp()
        pipe.hmset(self.meta_storage_key(id), mapping)

    def update_subscriber(self, subscriber, id, datetime):
        """Updates metadata for subscriber.

        Params:
            subscriber: a RedisBuffer instance
            id: the target id
            datetime: the new certification line of the subscriber
        """
        return self.client.hset(self.meta_storage_key(id),
                                subscriber.name, datetime.timestamp())

    def __query__(self, *columns, **kwargs):
        data_query = self.data_tract.query(*columns, **kwargs)
        meta_pipe = self.client.pipeline()
        if isinstance(data_query.id_range, rge.Level):
            id_range = [data_query.id_range.level]
        else:
            id_range = data_query.id_range
        meta_keys = ['lower', 'upper', 'certification']
        for id in id_range:
            meta_pipe.hmget(self.meta_storage_key(id), meta_keys)
        meta = meta_pipe.execute()
        metadata = [{k: pd.to_datetime(v, utc=True) for k, v in m.items()}
                    for m in meta]
        metaranges = [rge.time_range(m['lower'], m['upper']) for m in metadata]
        certificates = [m['certification'] for m in metadata]
        dt_range = rge.TimeInterval.intersection(data_query.dt_range,
                                                 *metaranges)
        certification = min(certificates)
        data_query.quargs['datetime'] = dt_range
        data_query.metadata = {'certification': certification}
        return data_query

    def __flush__(self, pipe, id, flushline):
        """Discards data behind flushline."""
        dt_range = rge.time_range(None, flushline)
        self.data_tract.__discard__(pipe, id, dt_range)

    def flush(self, id):
        """Discards from flushline based on certication & consumption."""
        metadata = self.metadata(id)
        certification = metadata.pop('certification')
        consumption = nxtime.MAX
        for k in metadata.keys():
            if k not in ['lower', 'upper']:
                v = metadata.pop(k)
                consumption = min((consumption, v))
        flushline = min((certification, consumption))
        pipe = self.client.pipeline()
        self.__flush__(pipe, id, flushline)

    def archive(self, sequence):
        return NotImplemented

    def update(self, sequence, old_certificate):
        """Updates the buffer by supplying a sequence.

        Data more recent than the old certificate is first discarded.
        Then the sequence data beyond the old certificate is appended.
        Newly certified data is archived.
        The flushline is computed and older data is discarded.
        Finally the metadata gets updated.
        """
        id = sequence.id
        pipe = self.client.pipeline()
        dt_range = rge.time_range(old_certificate + NS, None)
        self.data_tract.__discard__(pipe, id, dt_range)

        sequence = sequence[old_certificate + NS:]
        self.data_tract.__append__(pipe, sequence)

        self.archive(sequence[:sequence.certification])

        flushline = min((sequence.certification, sequence.consumption))
        self.__flush__(pipe, sequence.id, flushline)

        lower = flushline
        upper = sequence.dt_range.upper
        certification = sequence.certfication
        metadata = {'lower': lower,
                    'upper': upper,
                    'certification': certification}
        pipe.hmset(self.meta_storage_key(id), metadata)

        pipe.execute()
