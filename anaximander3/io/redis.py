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
from itertools import chain
import json

import pandas as pd
from redis import StrictRedis

from ..utilities import xprops, nxtime, nxrange as rge, functions as fun
from ..utilities.jsonmixin import serialize
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

    def __init__(self, store, title, register=True):
        super().__init__(store, title, register)

    def storage_key(self, id):
        """Returns the storage key given an application id."""
        return '#'.join((self.name, id))

    def __query__(self, *columns, **kwargs):
        return RedisDataQuery(self, *columns, **kwargs)

    def __insert__(self, pipe, id, recs, **kwargs):
        if not recs:
            return
        storage_key = self.name + '#' + str(id)
        data = {r.datetime.timestamp(): json.dumps(r.tabulated,
                                                   default=serialize)
                for r in recs}
        pipe.zadd(storage_key, *chain(*data.items()))

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
        updates = {r.datetime.timestamp(): r.tabulated for r in recs}
        min_score, max_score = min(updates), max(updates)
        extant = self.client.zrangebyscore(storage_key, min_score, max_score,
                                           withscores=True)
        extant = {s: data for data, s in extant}
        for score, tab in updates.items():
            try:
                elm = extant[score]
            except KeyError:
                pipe.zadd(storage_key, score,
                          json.dumps(tab, default=serialize))
            else:
                load = json.loads(elm)
                pipe.zrem(storage_key, elm)
                load.update(tab)
                pipe.zadd(storage_key, score,
                          json.dumps(load, default=serialize))

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
        data = json.loads(result[0])
        data.pop('id')
        data.pop('datetime')
        dict_ = {'data': data}
        dict_['index'] = idx
        dict_['schema'] = self.schema
        return rcd.Record.from_dict(dict_)

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

        for id in id_range:
            storage_key = self.tract.storage_key(id)
            pipe.zrevrangebyscore(storage_key, upper, lower,
                                  start, num, withscores=True)
            if self.schema.xindex:
                pipe.zrevrangebyscore(storage_key, lower,
                                      nxtime.MIN.timestamp(), 0, 1,
                                      withscores=True)
                id_sequence = chain(*zip(id_range, id_range))
            else:
                id_sequence = id_range

        sequence_results = pipe.execute()
        prev_id = None
        for id, sr in zip(id_sequence, sequence_results):
            for res, score in sr:
                if score == upper:
                    continue
                if score == lower:
                    if id == prev_id:
                        continue
                data = json.loads(res)
                idx = (data.pop('id'),
                       pd.to_datetime(data.pop('datetime'), utc=True))
                yield idx, {'data': data}
            prev_id = id


class RedisBuffer(RedisTract):
    """A specialized storage structure that combines data & metadata.

    The metadata is stored as a hash with the following structures:
        lower: timestamp  # lower bound of the buffer
        upper: timestamp  # upper bound of the buffer
        certification: timestamp  # certification line within buffer
        [subscriber title]: timestamp  # for each subscribing buffer, their
            own certification line, which determines data eviction.
    """

    def __init__(self, store, title, register=True):
        super().__init__(store, title, register)
        data_title = Title(title.name + '_data', title.schema)
        self.data_tract = RedisDataTract(store, data_title, register=False)

    def __get_data__(self, pipe, id, lower=None):
        dt_range = rge.time_range((lower, None))
        lower, upper = (t.timestamp() for t in dt_range)
        storage_key = self.tract.storage_key(id)
        pipe.zrevrangebyscore(storage_key, upper, lower, withscores=True)

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
        data = [json.loads(d) for d in data_]
        df = pd.DataFrame(data)
        return dtl.Sequence(df, schema=self.schema, id_range=id, **metadata)

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
