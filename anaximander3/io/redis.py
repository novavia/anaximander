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

import abc
from collections import defaultdict
from functools import partial
from itertools import chain, islice
import json

import pandas as pd
from redis import StrictRedis

from ..utilities import nxtime, nxrange as rge, functions as fun
from ..utilities.jsonmixin import serialize
from ..data import records as rcd, datalogs as dtl, nxschema as sch
from .store import Store, Title, Tract, DataTract, Query, QueryException, \
    ReadException, WriteException

__all__ = ['RedisStore', 'RedisTract', 'RedisDataQuery', 'RedisQueryException',
           'RedisInsertException']

NS = pd.Timedelta(nanoseconds=1)


def dtget(datetime, default=pd.NaT):
    """Utility that returns datetime if not None or NaT."""
    if datetime in (None, pd.NaT):
        return default
    return datetime


def take(n, iterable):
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))

# =============================================================================
# Tract implementation
# =============================================================================


class RedisInsertException(WriteException):
    pass


class RedisTract(Tract):

    @property
    def table_id(self):
        return self.name

    @property
    def client(self):
        return self.store.io

    def nxpipe(self, *args, **kwargs):
        return NxPipe(self.client, *args, **kwargs)

    def __exists__(self):
        """Tests existence of the resource. Must return True or False."""
        return True

    def __empty__(self):
        """Tests whether the resource is empty or not."""
        for k in self.client.keys(self.name + '*'):
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
        keys = self.client.keys(self.name + '*')
        if keys:
            self.client.delete(*keys)

    def migrate(self, store, batch_size=100):
        """Migrates a tract to the supplied destination store."""
        keys = self.client.keys(self.name + '*')
        for batch in fun.batch(keys, batch_size):
            org_pipe = self.client.pipeline()
            for key in batch:
                org_pipe.dump(key)
            values = org_pipe.execute()
            dst_pipe = store.io.pipeline()
            for key, value in zip(batch, values):
                if value is not None:
                    dst_pipe.restore(key, 0, value, replace=True)
            dst_pipe.execute()


class NxPipe:
    """Wrapper for Redis Pipe that applies transformations to data."""

    def __init__(self, store, *args, **kwargs):
        self._store = store
        self._pipe = store.io.pipeline(*args, **kwargs)
        self._callbacks = []

    @property
    def store(self):
        return self._store

    def execute(self):
        rval = []
        results = iter(self._pipe.execute())
        for callback, cardinal in self._callbacks:
            if cardinal is None:
                rval.append(callback(next(results)))
            else:
                rval.append(callback(*take(cardinal, results)))
        return [r for r in rval if r is not None]

    def pipe(self, callback=None, cardinal=None):
        if not cardinal == 0:
            callback = fun.get(callback, lambda *a: a)
            if callback is not False:
                self._callbacks.append((callback, cardinal))
        return self._pipe

    def record(self, tract, *idx, **kwargs):
        tract.record(*idx, _nxpipe=self, **kwargs)

    def records(self, tract, *idxs, keyerrors=True, **kwargs):
        for idx in idxs:
            tract.record(idx, _nxpipe=self, _raise=keyerrors, **kwargs)

    def query(self, tract, *columns, limit=None, kind='data', **quargs):
        query_ = tract.query(*columns, **quargs)
        if kind == 'data':
            query_.data(limit=limit, _nxpipe=self)
        elif kind == 'first':
            query_.first(_nxpipe=self)
        elif kind == 'fetch':
            query_.fetch(_nxpipe=self)

    def insert(self, tract, *records, validate=True, **kwargs):
        tract.insert(*records, validate=validate, _nxpipe=self, **kwargs)

    def append(self, tract, logs, **kwargs):
        tract.append(logs, _nxpipe=self, **kwargs)

    def discard(self, tract, *ids, datetime=(None, None), **kwargs):
        tract.discard(*ids, datetime=datetime, _nxpipe=self, **kwargs)

    def setup(self, buffer, id):
        buffer.setup(id, _nxpipe=self)

    def teardown(self, buffer, id):
        buffer.teardown(id, _nxpipe=self)

    def sequence(self, tract, id, **kwargs):
        tract.sequence(id, _nxpipe=self, **kwargs)

    def write(self, buffer, sequence, old_certificate=None):
        buffer.write(sequence, old_certificate=old_certificate, _nxpipe=self)

    def update(self, buffer, sequence, old_metadata=None):
        buffer.update(sequence, old_metadata=old_metadata, _nxpipe=self)

    def setup_subscriber(self, buffer, subscriber, id):
        buffer.setup_subscriber(subscriber, id, _nxpipe=self)

    def update_subscriber(self, buffer, subscriber, id, datetime):
        buffer.update_subscriber(subscriber, id, datetime, _nxpipe=self)

    def metadata(self, buffer, id):
        buffer.metadata(id, _nxpipe=self)

    def data(self, buffer, id, lower=None, upper=None):
        buffer.metadata(id, _nxpipe=self, lower=lower, upper=upper)


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
        data = [(r.datetime.timestamp(), json.dumps(r.to_data_dict(),
                                                    default=serialize))
                for r in recs]
        pipe.zadd(storage_key, *chain(*data))

    def insert(self, *records, validate=True, _nxpipe=None, **kwargs):
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

        pipe = _nxpipe.pipe(cardinal=len(rmap)) if _nxpipe\
            else self.client.pipeline()
        for id, recs in rmap.items():
            self.__insert__(pipe, id, recs, **kwargs)
        if _nxpipe is None:
            return pipe.execute()

    def __append__(self, pipe, logs, **kwargs):
        rmap = defaultdict(list)
        for record in list(logs):
            rmap[record.id].append(record)
        for id, recs in rmap.items():
            self.__insert__(pipe, id, recs, **kwargs)

    def append(self, logs, _nxpipe=None, **kwargs):
        cardinal = len({r.id for r in list(logs)})
        pipe = _nxpipe.pipe(cardinal=cardinal) if _nxpipe\
            else self.client.pipeline()
        super().append(logs, pipe=pipe, **kwargs)
        if _nxpipe is None:
            return pipe.execute()

    def __update__(self, pipe, id, recs, **kwargs):
        extant = {r.idx: r for r in self.records(*[r.idx for r in recs],
                                                 keyerrors=False)}
        removals = []
        appends = []
        updates = []
        for r in recs:
            try:
                old_rec = extant[r.idx]
                assert old_rec == r
            except KeyError:
                appends.append(r)
            except AssertionError:
                removals.append(old_rec)
                updates.append((old_rec, r))
        storage_key = self.name + '#' + str(id)
        for r in removals:
            data = json.dumps(r.to_data_dict(), default=serialize)
            pipe.zrem(storage_key, data)
        self.__insert__(pipe, id, appends)
        for old, new in updates:
            old_data = old.to_data_dict()
            new_data = new.to_data_dict()
            old_data['data'].update(new_data['data'])
            score = old.datetime.timestamp()
            pipe.zadd(storage_key, score,
                      json.dumps(old_data, default=serialize))

    def update(self, *records, _nxpipe=None, **kwargs):
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

        pipe = _nxpipe.pipe(cardinal=len(rmap)) if _nxpipe\
            else self.client.pipeline()
        for id, recs in rmap.items():
            self.__update__(pipe, id, recs, **kwargs)
        if _nxpipe is None:
            return pipe.execute()

    def __delete__(self, pipe, record, **kwargs):
        storage_key = self.name + '#' + str(record.id)
        data = json.dumps(record.to_data_dict(), default=serialize)
        pipe.zrem(storage_key, data)

    def delete(self, *idxs, _nxpipe=None, **kwargs):
        """Deletes the specified rows based on their index."""
        records = self.records(*idxs, **kwargs)
        pipe = _nxpipe.pipe(cardinal=len(records)) if _nxpipe\
            else self.client.pipeline()
        for rec in records:
            self.__delete__(pipe, rec, **kwargs)
        if _nxpipe is None:
            return pipe.execute()

    def __discard__(self, pipe, id, dt_range, **kwargs):
        """Deletes ranges of rows within dt_range for a given id."""
        storage_key = self.name + '#' + str(id)
        if isinstance(dt_range, rge.EmptyTimeInterval):
            lower, upper = 0, 0
        else:
            lower, upper = (t.timestamp() for t in dt_range)
        # Remove a nanosecond as a hack to exclude upper bound.
        upper -= 1e-9
        pipe.zremrangebyscore(storage_key, lower, upper)

    def discard(self, *ids, datetime=(None, None), _nxpipe=None, **kwargs):
        """Deletes ranges of rows within dt_range for specified ids."""
        dt_range = rge.time_range(datetime)
        pipe = _nxpipe.pipe(cardinal=len(ids)) if _nxpipe\
            else self.client.pipeline()
        for id in ids:
            self.__discard__(pipe, id, dt_range, **kwargs)
        if _nxpipe is None:
            return pipe.execute()

    def __record__(self, pipe, idx, **kwargs):
        id, datetime, *_ = idx
        storage_key = self.name + '#' + str(id)
        score = datetime.timestamp()
        pipe.zrangebyscore(storage_key, score, score)

    def _record(self, idx, result, _raise=True):
        """Primitive for record and records."""
        if not result:
            if _raise:
                msg = f"No row found with index {idx}."
                raise KeyError(msg)
            else:
                return None
        for res in result:
            dict_ = json.loads(res)
            ridx = dict_['index']
            ridx[1] = pd.to_datetime(ridx[1], utc=True)
            if tuple(ridx) == idx:
                break
            else:
                continue
        else:
            if _raise:
                msg = f"No row found with index {idx}."
                raise KeyError(msg)
            else:
                return None
        dict_['schema'] = self.schema
        return rcd.Record.from_dict(dict_)

    def record(self, *idx, _nxpipe=None, _raise=True, **kwargs):
        """Returns a record from its index."""
        if len(idx) == 1:
            idx = idx[0]
        id, datetime, *_ = idx
        idx = (id, pd.to_datetime(datetime, utc=True)) + tuple(_)
        pipe = _nxpipe.pipe(partial(self._record, idx, _raise=_raise)) \
            if _nxpipe else self.client.pipeline()
        self.__record__(pipe, idx, **kwargs)
        if _nxpipe is None:
            result = pipe.execute()[0]
            return self._record(idx, result, _raise=_raise)

    def records(self, *idxs, keyerrors=True, **kwargs):
        """Returns a records iterator from their index."""
        pipe = self.client.pipeline()
        for idx in idxs:
            self.__record__(pipe, idx, **kwargs)
        results = pipe.execute()
        for idx, result in zip(idxs, results):
            record = self._record(idx, result, _raise=keyerrors)
            if record is None:
                continue
            yield record


class RedisQueryException(QueryException):
    pass


class RedisDataQuery(Query):

    def __init__(self, tract, *columns, **quargs):
        super().__init__(tract, *columns, **quargs)
        if 'id' not in quargs:
            msg = "A Redis Query must specify an id range."
            raise RedisQueryException(msg)

    def _callback(self, kind='data'):
        id_range = self.id_range.levels
        if self.schema.xindex:
            cardinal = 2 * len(id_range)
        else:
            cardinal = len(id_range)
        if kind == 'data':
            return (lambda *r: Query.data(self, self._data(r)), cardinal)
        elif kind == 'first':
            return (lambda *r: Query.first(self, self._data(r)), cardinal)
        elif kind == 'fetch':
            return (lambda *r: iter(self._data(r)), cardinal)

    def first(self, _nxpipe=None, **kwargs):
        kwargs['limit'] = 1
        pipe = _nxpipe.pipe(*self._callback('first')) if _nxpipe\
            else self.tract.client.pipeline()
        self.__fetch__(pipe=pipe, **kwargs)
        if _nxpipe is None:
            results = self._data(pipe.execute())
            return super().first(_raw=results, **kwargs)

    def data(self, _nxpipe=None, **kwargs):
        pipe = _nxpipe.pipe(*self._callback('data')) if _nxpipe\
            else self.tract.client.pipeline()
        self.__fetch__(pipe=pipe, **kwargs)
        if _nxpipe is None:
            results = self._data(pipe.execute())
            return super().data(_raw=results, **kwargs)

    def fetch(self, _nxpipe=None, **kwargs):
        """Returns an iterator of query results, as raw data."""
        pipe = _nxpipe.pipe(*self._callback('fetch')) if _nxpipe\
            else self.tract.client.pipeline()
        self.__fetch__(pipe=pipe, **kwargs)
        if _nxpipe is None:
            results = self._data(pipe.execute())
            for index, data in results:
                yield index, data

    def __fetch__(self, pipe, limit=None, **kwargs):
        """Fetch primitive.

        Note that queries on sequential keys are closed on the left side
        and open on the right side.
        """
        id_range = self.id_range.levels
        if isinstance(self.dt_range, rge.EmptyTimeInterval):
            lower, upper = 0, 0
        else:
            lower, upper = (t.timestamp() for t in self.dt_range)
        limit = int(fun.get(limit, 1e5))
        start, num = 0, limit

        for id in id_range:
            storage_key = self.tract.storage_key(id)
            pipe.zrevrangebyscore(storage_key, upper, lower,
                                  start, num, withscores=True)
            if self.schema.xindex:
                pipe.zrevrangebyscore(storage_key, lower,
                                      nxtime.MIN.timestamp(), 0, 1,
                                      withscores=True)

    def _data(self, results):
        if isinstance(self.dt_range, rge.EmptyTimeInterval):
            lower, upper = 0, 0
        else:
            lower, upper = (t.timestamp() for t in self.dt_range)
        id_range = self.id_range.levels
        if self.schema.xindex:
            id_sequence = chain(*zip(id_range, id_range))
        else:
            id_sequence = id_range

        rval = []
        prev_id = None
        for id, sr in zip(id_sequence, results):
            for res, score in sr:
                dict_ = json.loads(res)
                idx = dict_.pop('index')
                if score == upper:
                    continue
                if id == prev_id:
                    if score == lower:
                        continue
                    elif isinstance(self.schema, sch.SessionLogsSchema):
                        try:
                            duration = pd.Timedelta(dict_['data']['duration'])
                            start = pd.to_datetime(idx[1], utc=True)
                            if start + duration <= self.dt_range.lower:
                                continue
                        except (KeyError, ValueError, TypeError):
                            continue
                rval.append((idx, dict_))
            prev_id = id
        return rval


class BufferQuery(RedisDataQuery):
    """Specialized query type that doesn't aggregate results across ids."""

    def __init__(self, tract, *columns, **quargs):
        super().__init__(tract, *columns, **quargs)
        try:
            assert isinstance(tract, RedisBuffer)
        except AssertionError:
            msg = "A BufferQuery requires a  RedisBuffer tract."
            raise RedisQueryException(msg)
        self.data_query = tract.data_tract.query(**self.quargs)

    def _callback(self, kind='data'):
        if kind == 'drop':
            return (False, None)
        id_range = self.id_range.levels
        n = len(id_range)
        if self.schema.xindex:
            cardinal = 3 * n
        else:
            cardinal = 2 * n
        if kind == 'data':

            def callback(*results):
                metadata = self._metadata(results[0:n])
                logs = Query.data(self.data_query,
                                  self.data_query._data(results[n:]))
                return self._post_process(logs, metadata)

        elif kind == 'first':

            def callback(*results):
                raw_logs = self.data_query._data(results[n:])
                return Query.first(self, _raw=raw_logs)

        elif kind == 'fetch':

            def callback(*results):
                data = self.data_query._data(results[n:])
                return iter(data)

        return (callback, cardinal)

    def _metadata(self, results):
        """Fetches metadata for self."""
        id_range = self.id_range.levels
        meta_keys = ['lower', 'upper', 'certification']
        protometa = [{k: pd.to_datetime(v.decode(), utc=True)
                      for k, v in zip(meta_keys, r) if v is not None}
                     for r in results]
        metadata = {'dt_ranges': {id: rge.EmptyTimeInterval()
                                  for id in self.id_range.levels},
                    'certificates': {id: pd.NaT
                                     for id in self.id_range.levels},
                    'consumptions': {id: nxtime.MAX
                                     for id in self.id_range.levels}}
        for id, m in zip(id_range, protometa):
            metadata['dt_ranges'][id] = rge.time_range(m.pop('lower', pd.NaT),
                                                       m.pop('upper', pd.NaT))
            metadata['certificates'][id] = dtget(m.pop('certification',
                                                       pd.NaT))
            consumption = nxtime.MAX
            for k in list(m.keys()):
                v = protometa.pop(k)
                consumption = min(v, consumption, v)
            metadata['consumptions'][id] = consumption
        return metadata

    def __metadata__(self, pipe):
        id_range = self.id_range.levels
        meta_keys = ['lower', 'upper', 'certification']
        for id in id_range:
            pipe.hmget(self.tract.meta_storage_key(id), meta_keys)

    def first(self, _nxpipe=None, **kwargs):
        id_range = self.id_range.levels
        n = len(id_range)
        kwargs['limit'] = 1
        pipe = _nxpipe.pipe(*self._callback('first')) if _nxpipe\
            else self.tract.client.pipeline()
        self.__fetch__(pipe=pipe, **kwargs)
        if _nxpipe is None:
            raw = pipe.execute()
            results = self.data_query._data(raw[n:])
            return Query.first(self, _raw=results, **kwargs)

    def data(self, _nxpipe=None, **kwargs):
        id_range = self.id_range.levels
        n = len(id_range)
        pipe = _nxpipe.pipe(*self._callback('data')) if _nxpipe\
            else self.tract.client.pipeline()
        self.__fetch__(pipe=pipe, **kwargs)
        if _nxpipe is None:
            raw = pipe.execute()
            metadata = self._metadata(raw[0:n])
            logs = Query.data(self.data_query,
                              self.data_query._data(raw[n:]))
            return self._post_process(logs, metadata)

    def fetch(self, _nxpipe=None, **kwargs):
        """Returns an iterator of query results, as raw data."""
        id_range = self.id_range.levels
        n = len(id_range)
        pipe = _nxpipe.pipe(*self._callback('fetch')) if _nxpipe\
            else self.tract.client.pipeline()
        self.__fetch__(pipe=pipe, **kwargs)
        if _nxpipe is None:
            raw = pipe.execute()
            results = self.data_query._data(raw[n:])
            for index, data in results:
                yield index, data

    def __fetch__(self, pipe, limit=None, **kwargs):
        """Returns an iterator of row indexes and data."""
        self.__metadata__(pipe=pipe)
        self.data_query.__fetch__(pipe, limit=limit, **kwargs)

    def _post_process(self, logs, metadata):
        """Combines data and metadata."""
        if isinstance(self.id_range, rge.Levels):
            pp_data = []
            for id in self.id_range:
                pp_sequence = logs[id]
                pp_metadata = pp_sequence.metadata
                pp_metadata['dt_range'] = metadata['dt_ranges'][id] &\
                    self.dt_range
                pp_metadata['certification'] = metadata['certificates'][id]
                pp_metadata['consumption'] = metadata['consumptions'][id]
                sequence = dtl.DataSequence(pp_sequence.data,
                                            schema=self.schema,
                                            **pp_metadata)
                pp_data.append(sequence)
            return pp_data
        else:
            id = self.id_range.level
            pp_metadata = logs.metadata
            pp_metadata['dt_range'] = metadata['dt_ranges'][id] &\
                self.dt_range
            pp_metadata['certification'] = metadata['certificates'][id]
            pp_metadata['consumption'] = metadata['consumptions'][id]
            sequence = dtl.DataSequence(logs.data,
                                        schema=self.schema,
                                        **pp_metadata)
            return sequence


class NotInStoreException(ReadException):
    """Exception raised on buffers if id is not found in store."""
    pass


class NoArchive(WriteException):
    """Exception raised when attempting to access non-existing archive."""
    pass


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

    def setup(self, id, _nxpipe=None, **kwargs):
        """Setup storage for supplied id."""
        now = nxtime.now()
        sequence = dtl.DataSequence(schema=self.schema,
                                    id_range=id,
                                    dt_range=(now, now))
        self.write(sequence, _nxpipe=_nxpipe)

    def teardown(self, id, _nxpipe=None):
        """Discards storage for supplied id."""
        pipe = _nxpipe.pipe(cardinal=2) if _nxpipe else self.client.pipeline()
        pipe.delete(self.data_tract.storage_key(id))
        pipe.delete(self.meta_storage_key(id))
        if _nxpipe is None:
            return pipe.execute()

    def __get_data__(self, pipe, id, lower=None, upper=None):
        dt_range = rge.time_range((lower, upper))
        lower, upper = (t.timestamp() for t in dt_range)
        storage_key = self.data_tract.storage_key(id)
        pipe.zrevrangebyscore(storage_key, upper, lower)
        if self.schema.xindex:
            pipe.zrevrangebyscore(storage_key, lower,
                                  nxtime.MIN.timestamp(), 0, 1)

    def _data(self, *results):
        """Primitive for data."""
        records = []
        for res in chain(*results):
            dict_ = json.loads(res)
            dict_['schema'] = self.schema
            records.append(rcd.Record.from_dict(dict_))
        return records

    def data(self, id, lower=None, upper=None, _nxpipe=None):
        """Queries and returns records for supplied id.

        Note that this doesn't involve metadata. Use sequence to download
        a full DataSequence with metadata.
        """
        if self.schema.xindex:
            card = 2
        else:
            card = 1
        pipe = _nxpipe.pipe(self._data, card) if _nxpipe\
            else self.client.pipeline()
        self.__get_data__(pipe, id, lower=lower, upper=upper)
        if _nxpipe is None:
            return self._data(*pipe.execute())

    def _sequence(self, metadata, *data, id=id, lower=None, upper=None):
        if not metadata:
            raise NotInStoreException()
        meta_ = {k.decode(): pd.to_datetime(v.decode(), utc=True)
                 for k, v in metadata.items()}
        lower_ = meta_.pop('lower', pd.NaT)
        upper_ = meta_.pop('upper', pd.NaT)
        if lower is not None:
            lower_ = max(lower, lower_)
        if upper is not None:
            upper_ = min(upper, upper_)
        dt_range = rge.time_range(lower_, upper_)
        consumption = nxtime.MAX
        for k in list(meta_.keys()):
            if k != 'certification':
                v = meta_.pop(k)
                consumption = min(v, consumption)
        meta_['dt_range'] = dt_range
        meta_['consumption'] = consumption
        data_ = []
        for res in chain(*data):
            dict_ = json.loads(res)
            dt = dict_['data']
            dt['id'], dt['datetime'] = dict_['index']
            data_.append(dt)
        df = pd.DataFrame(data_)
        if df.empty:
            df = None
        return dtl.DataSequence(df, schema=self.schema,
                                id_range=id, **meta_)

    def sequence(self, id, lower=None, upper=None, _nxpipe=None):
        """Fetches data for id, optionally above lower datetime."""
        callback = partial(self._sequence, id=id, lower=lower, upper=upper)
        if self.schema.xindex:
            card = 3
        else:
            card = 2
        if _nxpipe:
            pipe = _nxpipe.pipe(callback, card)
        else:
            pipe = self.client.pipeline()
        self.__get_metadata__(pipe, id)
        self.__get_data__(pipe, id, lower=lower, upper=upper)
        if _nxpipe is None:
            metadata, *data = pipe.execute()
            return self._sequence(metadata, *data, id=id,
                                  lower=lower, upper=upper)

    def meta_storage_key(self, id):
        """Returns the storage key for metadata."""
        return '#'.join((self.name + '_meta', id))

    def __get_metadata__(self, pipe, id):
        pipe.hgetall(self.meta_storage_key(id))

    def _metadata(self, results):
        if not results:
            raise NotInStoreException()
        return {k.decode(): pd.to_datetime(v.decode(), utc=True)
                for k, v in results.items()}

    def metadata(self, id, _nxpipe=None):
        """Queries and returns the metadata for supplied id."""
        pipe = _nxpipe.pipe(self._metadata) if _nxpipe\
            else self.client.pipeline()
        self.__get_metadata__(pipe, id)
        if _nxpipe is None:
            return self._metadata(pipe.execute()[0])

    def __query__(self, *columns, **kwargs):
        return BufferQuery(self, *columns, **kwargs)

    @abc.abstractmethod
    def flushline(self, id, sequence):
        return nxtime.MIN

    @abc.abstractmethod
    def write(self, sequence, old_certificate=None, _nxpipe=None):
        super().write(sequence, old_certificate=old_certificate,
                      _nxpipe=_nxpipe)


class RedisApplicationBuffer(RedisBuffer):

    def __init__(self, store, title, depth, register=True):
        super().__init__(store, title, register)
        self.depth = pd.Timedelta(depth)

    def setup(self, id, when=None, _nxpipe=None):
        """Sets up storage for supplied id."""
        if when is None:
            when = nxtime.now()
        else:
            when = pd.to_datetime(when, utc=True)
        try:
            sequence = self._load_from_archive(id, when=when)
            sequence.certify(sequence.dt_range.upper)
        except NoArchive:
            sequence = dtl.DataSequence(schema=self.schema,
                                        id_range=id,
                                        dt_range=(when, when))
        self.write(sequence, _nxpipe=_nxpipe)

    def _load_from_archive(self, id, when):
        archive_store = self.store.archive
        if archive_store is None:
            msg = "Cannot retrieve from non-existing archive."
            raise NoArchive(msg)
        else:
            try:
                archive_tract = archive_store[self.title]
            except KeyError:
                msg = f"Cannot retrieve data from {archive_store} " + \
                      f"because it doesn't feature a table for {self.title}."
                raise NoArchive(msg)
        start = when - self.depth
        return archive_tract.query(id=id, datetime=(start, when)).data()

    def flushline(self, id, sequence, old_upper=None):
        if old_upper is None:
            old_metadata = self.metadata(id)
            old_upper = old_metadata['upper']
        upper = max(old_upper, sequence.dt_range.upper)
        if pd.isna(upper):
            return nxtime.MIN
        else:
            return upper - self.depth

    def mock(self, id, _nxpipe=None):
        """Returns an empty sequence with self.metadata attached."""
        def callback(results):
            metadata = self._metadata(results)
            return dtl.DataSequence(schema=self.schema, id_range=id,
                                    **metadata)
        if _nxpipe:
            pipe = _nxpipe.pipe(callback)
        else:
            pipe = self.client.pipeline()
        self.__get_metadata__(pipe, id)
        if _nxpipe is None:
            results = pipe.execute()
            return callback(results)

    def write(self, sequence, old_certificate=None, _nxpipe=None):
        """Writes the buffer by supplying a sequence.

        Overwrites data with sequence from old certificate on.
        Finally the metadata gets updated.
        """
        if not isinstance(sequence, dtl.DataSequence):
            msg = f"Buffer write requires a DataSequence instance, not a " + \
                  f"{type(sequence)} instance."
            raise TypeError(msg)
        if not isinstance(sequence.schema, type(self.schema)):
            msg = f"Incorrect data schema supplied to {self}."
            raise ValueError(msg)
        write_range = rge.time_range((sequence.dt_range.upper - self.depth,
                                      sequence.dt_range.upper))
        sequence = sequence[slice(*write_range)]
        id = sequence.id
        cardinal = 2 if sequence.empty else 3
        pipe = _nxpipe.pipe(cardinal=cardinal) if _nxpipe\
            else self.client.pipeline()
        pipe.delete(self.data_tract.storage_key(id))
        self.data_tract.__append__(pipe, sequence)
        lower, upper = sequence.dt_range
        metadata = {'lower': str(lower),
                    'upper': str(upper),
                    'certification': str(sequence.certification)}
        pipe.hmset(self.meta_storage_key(id), metadata)
        if _nxpipe is None:
            return pipe.execute()

    def update(self, id, sequence, old_metadata=None, _nxpipe=None):
        """Updates the buffer from old certificate on."""
        if old_metadata is None:
            if _nxpipe is not None:
                msg = "Cannot pipe an update without extant metadata."
                raise ValueError(msg)
            else:
                old_metadata = self.metadata(id)
                lower, upper = old_metadata['lower'], old_metadata['upper']
                old_metadata['dt_range'] = rge.time_range((lower, upper))
        update_line = old_metadata['certification']
        if pd.isna(update_line):
            return self.write(sequence, _nxpipe=_nxpipe)
        update_range = rge.time_range((update_line, None))
        flushline = self.flushline(id, sequence,
                                   old_metadata['dt_range'].upper)
        flushrange = rge.time_range((None, flushline))
        data = sequence[update_line:]
        cardinal = 3 if data.empty else 4
        pipe = _nxpipe.pipe(cardinal=cardinal) if _nxpipe\
            else self.client.pipeline()
        self.data_tract.__discard__(pipe, id, update_range)
        self.data_tract.__discard__(pipe, id, flushrange)
        self.data_tract.__append__(pipe, data)
        lower = max(old_metadata['dt_range'].lower, flushline)
        upper = max(old_metadata['dt_range'].upper, sequence.dt_range.upper)
        certificate = max(update_line, sequence.certification)
        metadata = {'lower': str(lower),
                    'upper': str(upper),
                    'certification': str(certificate)}
        pipe.hmset(self.meta_storage_key(id), metadata)
        if _nxpipe is None:
            return pipe.execute()


class RedisProcessBuffer(RedisBuffer):

    def teardown(self, id, _nxpipe=None):
        """Discards storage for supplied id.

        For state logs we archive an nan stitch.
        """
        if isinstance(self.schema, sch.StateLogsSchema):
            try:
                sequence = self.sequence(id)
                cert = sequence.certification
                assert not pd.isna(cert)
                record = sequence[cert].nulled_copy(cert)
                archive_store = self.store.archive
                archive_tract = archive_store[self.title]
                archive_tract.update(record)
            except:
                pass
        super().teardown(id, _nxpipe=_nxpipe)

    def setup_subscriber(self, subscriber, id, _nxpipe=None):
        """Sets up metadata for subscriber.

        Params:
            subscriber: a RedisBuffer instance
            id: the target id
        """
        self.update_subscriber(subscriber, id, pd.NaT, _nxpipe=None)

    def update_subscriber(self, subscriber, id, datetime, _nxpipe=None):
        """Updates metadata for subscriber.

        Params:
            subscriber: a RedisBuffer instance
            id: the target id
            datetime: the new certification line of the subscriber
        """
        pipe = _nxpipe.pipe() if _nxpipe else self.client.pipeline()
        pipe.hset(self.meta_storage_key(id), subscriber.name, str(datetime))
        if _nxpipe is None:
            return pipe.execute()

    def archive(self, sequence, old_certificate=None):
        """Archives sequence against old certificate."""
        archive_store = self.store.archive
        if archive_store is None:
            msg = f"Cannot archive {sequence}  " + \
                  f"because no archive store is setup."
            raise NoArchive(msg)
        else:
            try:
                archive_tract = archive_store[self.title]
            except KeyError:
                msg = f"Cannot archive {sequence} into {archive_store} " + \
                      f"because it doesn't feature a table for {self.title}."
                raise NoArchive(msg)
        archive_bound = sequence[old_certificate:sequence.certification]
        archive_tract.append(archive_bound)

    def flushline(self, id, sequence):
        certification = dtget(sequence.certification, nxtime.MIN)
        consumption = dtget(sequence.consumption, nxtime.MIN)
        return min((certification, consumption))

    def write(self, sequence, old_certificate=None, _nxpipe=None):
        """Writes the buffer by supplying a sequence.

        Overwrites data with sequence.
        Newly certified data is archived.
        Finally the metadata gets updated.
        """
        if not isinstance(sequence, dtl.DataSequence):
            msg = f"Buffer write requires a DataSequence instance, not a " + \
                  f"{type(sequence)} instance."
            raise TypeError(msg)
        if not isinstance(sequence.schema, type(self.schema)):
            msg = f"Incorrect data schema supplied to {self}."
            raise ValueError(msg)
        id = sequence.id
        cardinal = 2 if sequence.empty else 3
        pipe = _nxpipe.pipe(cardinal=cardinal) if _nxpipe\
            else self.client.pipeline()
        pipe.delete(self.data_tract.storage_key(id))
        try:
            self.archive(sequence, old_certificate)
        except NoArchive:
            pass
        flushline = self.flushline(sequence.id, sequence)
        sequence = sequence[flushline:]
        self.data_tract.__append__(pipe, sequence)
        lower, upper = sequence.dt_range
        metadata = {'lower': str(lower),
                    'upper': str(upper),
                    'certification': str(sequence.certification)}
        pipe.hmset(self.meta_storage_key(id), metadata)
        if _nxpipe is None:
            return pipe.execute()


class RedisStore(Store):

    def __interface__(self, host, port, password):
        return StrictRedis(host=host, port=port, password=password)

    def migrate(self, destination):
        """Requires a destination store."""
        for title, tract in self.items():
            destination.tract(tract.title)
            tract.migrate(destination)

    def __repr__(self):
        conn_kwargs = self.io.connection_pool.connection_kwargs
        h, p, db = conn_kwargs['host'], conn_kwargs['port'], conn_kwargs['db']
        return f'<{type(self).__name__} host:{h} port:{p} db:{db}>'


class RedisArchive(RedisStore):
    __tract__ = RedisDataTract
    __role__ = 'archive'


class RedisApplicationStore(RedisStore):
    __tract__ = RedisApplicationBuffer
    __role__ = 'buffer'

    def __interface__(self, host, port, password, archive=None):
        """Archive is an optional archive store."""
        self.archive = archive
        return StrictRedis(host=host, port=port, password=password)

    def tract(self, title, depth):
        """Instantiates a tract for self."""
        return self.__tract__(self, title, depth)


class RedisProcessStore(RedisStore):
    __tract__ = RedisProcessBuffer
    __role__ = 'process'

    def __interface__(self, host, port, password, archive=None):
        """Archive is an optional archive store."""
        self.archive = archive
        return StrictRedis(host=host, port=port, password=password)
