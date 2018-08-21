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
from itertools import chain
import json

import pandas as pd
from redis import StrictRedis

from ..utilities import nxtime, nxrange as rge, functions as fun, xprops
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
        rmap = defaultdict(list)
        for record in list(logs):
            rmap[record.id].append(record)
        for id, recs in rmap.items():
            self.__insert__(pipe, id, recs, **kwargs)

    def append(self, logs, **kwargs):
        pipe = self.client.pipeline()
        kwargs.setdefault('pipe', pipe)
        super().append(logs, **kwargs)
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

    def __delete__(self, pipe, record, **kwargs):
        storage_key = self.name + '#' + str(record.id)
        data = json.dumps(record.to_data_dict(), default=serialize)
        pipe.zrem(storage_key, data)

    def delete(self, *idxs, **kwargs):
        """Deletes the specified rows based on their index."""
        records = self.records(*idxs, **kwargs)
        pipe = self.client.pipeline()
        for rec in records:
            self.__delete__(pipe, rec, **kwargs)
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
        id, datetime, *_ = idx
        storage_key = self.name + '#' + str(id)
        score = datetime.timestamp()
        pipe.zrangebyscore(storage_key, score, score)

    def _record(self, idx, result):
        """Primitive for record and records."""
        if not result:
            msg = f"No row found with index {idx}."
            raise KeyError(msg)
        for r in result:
            dict_ = json.loads(r)
            ridx = dict_['index']
            ridx[1] = pd.to_datetime(ridx[1], utc=True)
            if tuple(ridx) == idx:
                break
            else:
                continue
        else:
            msg = f"No row found with index {idx}."
            raise KeyError(msg)
        dict_['schema'] = self.schema
        return rcd.Record.from_dict(dict_)

    def record(self, *idx, **kwargs):
        """Returns a record from its index."""
        if len(idx) == 1:
            idx = idx[0]
        id, datetime, *_ = idx
        idx = (id, pd.to_datetime(datetime, utc=True)) + tuple(_)
        pipe = self.client.pipeline()
        self.__record__(pipe, idx, **kwargs)
        result = pipe.execute()[0]
        return self._record(idx, result)

    def records(self, *idxs, keyerrors=True, **kwargs):
        """Returns a records iterator from their index."""
        pipe = self.client.pipeline()
        for idx in idxs:
            self.__record__(pipe, idx, **kwargs)
        results = pipe.execute()
        for idx, result in zip(idxs, results):
            try:
                yield self._record(idx, result)
            except KeyError:
                if not keyerrors:
                    continue
                else:
                    raise


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
        id_range = self.id_range.levels
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
                yield idx, dict_
            prev_id = id


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

    @xprops.cachedproperty
    def dt_ranges(self):
        return {id: rge.EmptyTimeInterval() for id in self.id_range.levels}

    @xprops.cachedproperty
    def certificates(self):
        return {id: pd.NaT for id in self.id_range.levels}

    def _metadata(self):
        """Fetches metadata for self."""
        meta_pipe = self.tract.client.pipeline()
        id_range = self.id_range.levels
        meta_keys = ['lower', 'upper', 'certification']
        for id in id_range:
            meta_pipe.hmget(self.tract.meta_storage_key(id), meta_keys)
        meta = meta_pipe.execute()
        metadata = [{k: pd.to_datetime(v.decode(), utc=True)
                     for k, v in zip(meta_keys, m) if v is not None}
                    for m in meta]
        for id, m in zip(id_range, metadata):
            self.dt_ranges[id] = rge.time_range(m.get('lower', pd.NaT),
                                                m.get('upper', pd.NaT))
            self.certificates[id] = dtget(m.get('certification'))

    def __fetch__(self, **kwargs):
        """Returns an iterator of row indexes and data."""
        self._metadata()
        return self.data_query.__fetch__(**kwargs)

    def data(self, **kwargs):
        """Returns the data."""
        super_data = super().data(**kwargs)
        if isinstance(self.id_range, rge.Levels):
            data = []
            for id in self.id_range:
                super_sequence = super_data[id]
                metadata = super_sequence.metadata
                metadata['dt_range'] = self.dt_ranges[id]
                metadata['certification'] = self.certificates[id]
                sequence = dtl.DataSequence(super_sequence.data,
                                            schema=self.schema,
                                            **metadata)
                data.append(sequence)
            return data
        else:
            id = self.id_range.level
            super_sequence = super_data
            metadata = super_sequence.metadata
            metadata['dt_range'] = self.dt_ranges[id]
            metadata['certification'] = self.certificates[id]
            sequence = dtl.DataSequence(super_sequence.data,
                                        schema=self.schema,
                                        **metadata)
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

    def setup(self, id):
        """Setup storage for supplied id."""
        now = nxtime.now()
        sequence = dtl.DataSequence(schema=self.schema,
                                    id_range=id,
                                    dt_range=(now, now))
        self.write(sequence)

    def teardown(self, id):
        """Discards storage for supplied id."""
        pipe = self.client.pipeline()
        pipe.delete(self.data_tract.storage_key(id))
        pipe.delete(self.meta_storage_key(id))
        pipe.execute()

    def __get_data__(self, pipe, id, lower=None):
        dt_range = rge.time_range((lower, None))
        lower, upper = (t.timestamp() for t in dt_range)
        storage_key = self.data_tract.storage_key(id)
        pipe.zrevrangebyscore(storage_key, upper, lower)

    def sequence(self, id, lower=None):
        """Fetches data for id, optionally above lower datetime."""
        if self.schema.xindex and lower is not None:
            msg = "Cannot fetch partial sequence on an xindex schema."
            raise ValueError(msg)
        pipe = self.client.pipeline()
        pipe.exists(self.meta_storage_key(id))
        self.__get_metadata__(pipe, id)
        self.__get_data__(pipe, id, lower=lower)
        id_exists, meta_, data_ = pipe.execute()
        if not id_exists:
            raise NotInStoreException()
        metadata = {k.decode(): pd.to_datetime(v.decode(), utc=True)
                    for k, v in meta_.items()}
        dt_range = rge.time_range(metadata.pop('lower', pd.NaT),
                                  metadata.pop('upper', pd.NaT))
        consumption = nxtime.MAX
        for k in list(metadata.keys()):
            if k != 'certification':
                v = metadata.pop(k)
                consumption = min((consumption, v))
        metadata['dt_range'] = dt_range
        metadata['consumption'] = consumption
        data = []
        for res in data_:
            dict_ = json.loads(res)
            dt = dict_['data']
            dt['id'], dt['datetime'] = dict_['index']
            data.append(dt)
        df = pd.DataFrame(data)
        if df.empty:
            df = None
        return dtl.DataSequence(df, schema=self.schema,
                                id_range=id, **metadata)

    def meta_storage_key(self, id):
        """Returns the storage key for metadata."""
        return '#'.join((self.name + '_meta', id))

    def __get_metadata__(self, pipe, id):
        pipe.hgetall(self.meta_storage_key(id))

    def metadata(self, id):
        """Queries and returns the metadata for supplied id."""
        pipe = self.client.pipeline()
        pipe.exists(self.meta_storage_key(id))
        self.__get_metadata__(pipe, id)
        id_exists, meta = pipe.execute()
        if not id_exists:
            raise NotInStoreException()
        return {k.decode(): pd.to_datetime(v.decode(), utc=True)
                for k, v in meta.items()}

    def __query__(self, *columns, **kwargs):
        return BufferQuery(self, *columns, **kwargs)

    @abc.abstractmethod
    def write(self, sequence, old_certificate=None):
        super().write(sequence)


class RedisApplicationBuffer(RedisBuffer):

    def __init__(self, store, title, depth, register=True):
        super().__init__(store, title, register)
        self.depth = pd.Timedelta(depth)

    def setup(self, id, when=None):
        """Sets up storage for supplied id."""
        if when is None:
            when = nxtime.now()
        else:
            when = pd.to_datetime(when, utc=True)
        try:
            sequence = self._load_from_archive(id, when=when)
        except NoArchive:
            sequence = dtl.DataSequence(schema=self.schema,
                                        id_range=id,
                                        dt_range=(when, when))
        self.write(sequence)

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

    def write(self, sequence, old_certificate=None):
        """Writes the buffer by supplying a sequence.

        Overwrites data with sequence.
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
        pipe = self.client.pipeline()
        pipe.delete(self.data_tract.storage_key(id))
        self.data_tract.__append__(pipe, sequence)
        lower, upper = sequence.dt_range
        metadata = {'lower': str(lower),
                    'upper': str(upper),
                    'certification': str(sequence.certification)}
        pipe.hmset(self.meta_storage_key(id), metadata)
        pipe.execute()


class RedisProcessBuffer(RedisBuffer):

    def teardown(self, id):
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
        super().teardown(id)

    def setup_subscriber(self, subscriber, id):
        """Sets up metadata for subscriber.

        Params:
            subscriber: a RedisBuffer instance
            id: the target id
        """
        self.update_subscriber(subscriber, id, pd.NaT)

    def update_subscriber(self, subscriber, id, datetime):
        """Updates metadata for subscriber.

        Params:
            subscriber: a RedisBuffer instance
            id: the target id
            datetime: the new certification line of the subscriber
        """
        return self.client.hset(self.meta_storage_key(id),
                                subscriber.name, str(datetime))

    def archive(self, sequence, old_certificate=None):
        """Archives sequence against old certificate."""
        archive_store = self.store.archive
        if archive_store is None:
            msg = f"Cannot archive {sequence}  " + \
                  f"because not archive store is setup."
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

    def write(self, sequence, old_certificate=None):
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
        pipe = self.client.pipeline()
        pipe.delete(self.data_tract.storage_key(id))
        try:
            self.archive(sequence, old_certificate)
        except NoArchive:
            pass
        certification = dtget(sequence.certification, nxtime.MIN)
        consumption = dtget(sequence.consumption, nxtime.MIN)
        flushline = min((certification, consumption))
        sequence = sequence[flushline:]
        self.data_tract.__append__(pipe, sequence)
        lower, upper = sequence.dt_range
        metadata = {'lower': str(lower),
                    'upper': str(upper),
                    'certification': str(sequence.certification)}
        pipe.hmset(self.meta_storage_key(id), metadata)
        pipe.execute()


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


class RedisApplicationStore(RedisStore):
    __tract__ = RedisApplicationBuffer

    def __interface__(self, host, port, password, archive=None):
        """Archive is an optional archive store."""
        self.archive = archive
        return StrictRedis(host=host, port=port, password=password)

    def tract(self, title, depth):
        """Instantiates a tract for self."""
        return self.__tract__(self, title, depth)


class RedisProcessStore(RedisStore):
    __tract__ = RedisProcessBuffer

    def __interface__(self, host, port, password, archive=None):
        """Archive is an optional archive store."""
        self.archive = archive
        return StrictRedis(host=host, port=port, password=password)
