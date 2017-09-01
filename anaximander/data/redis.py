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

from redis import StrictRedis

from ..utilities import functions as fun, nxattr, nxtime, xprops
from ..meta import prototype, metacharacter, cachedtypeproperty
from .schema import Schema
from .table import DataTable, DataQuery, DataQueryException, \
    DataTableWriteException

__all__ = ['RedisDataTable', 'RedisQuery', 'RedisQueryException',
           'RedisInsertException', 'client']


# =============================================================================
# Client factory
# =============================================================================


def client(host, port, password, max_connections):
    return StrictRedis(host=host, port=port, password=password,
                       max_connections=max_connections)

# =============================================================================
# Key generation
# =============================================================================


def scoremaker(schema):
    """Returns a scoring function from a schema.

    The scoring function is used to set the score of a record when inserting
    them into Redis ordered sets. The scoring startegy is set by passing
    a string value to the key declaration of a sequential field in schema
    definitions. It defaults to identity.
    """
    strategies = {'hash': lambda x: hash(x),
                  'timestamp': lambda x: int(1e6 * (nxtime.MAX_TIMESTAMP -
                                                    x.timestamp()))}
    seqkey_name = schema.seqkey
    if seqkey_name is None:
        return None
    seqkey_field = schema.fields[seqkey_name]
    return strategies.get(seqkey_field.key, lambda x: x)

# =============================================================================
# Table implementation
# =============================================================================


class RedisInsertException(DataTableWriteException):
    pass


@prototype
@nxattr.s(hash=False, repr=False)
class RedisDataTable(DataTable):
    schema = metacharacter(validate=fun.subcheck(Schema))
    instance = nxattr.ib(validator=nxattr.validators.instance_of(StrictRedis))
    name = nxattr.ib(validator=nxattr.validators.instance_of(str))
    # Data retention time, in seconds, defaulting to 1 hour, optionally None
    retention = nxattr.ib(3600, repr=False)
    # An optional maxrate to limit query size automatically
    # This functionality requires a rowcount method to be implemented on
    # the table's Schema
    maxrate = nxattr.ib(None, repr=False)

    @cachedtypeproperty
    def scoring(cls):
        """A scoring function for sequential keys."""
        return scoremaker(cls.schema)

    def __attrs_post_init__(self):
        tract = self.schema.tract
        if tract is not None:
            tract._redis = self

    @property
    def client(self):
        return self.instance

    @property
    def version(self):
        """The current version, which is time-dependent based on retention."""
        if self.retention is None:
            return 0
        now = int(nxtime.now().timestamp())
        return now // self.retention

    def key(self, record, version=None):
        """Returns the insertion key for a record."""
        version = str(fun.get(version, self.version))
        fields = '#'.join(str(getattr(record, k)) for k in self.schema.nskeys)
        return '#'.join((self.name, fields, version))

    def score(self, record):
        """Returns the 'score' of a record for ordered set insertion."""
        if self.scoring is None:
            return 1
        seqkey = getattr(record, self.schema.seqkey)
        return self.scoring(seqkey)

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
        pass

    def exists(self):
        return True

    def __remove__(self):
        pass

    def __query__(self, *fields, **kwargs):
        return RedisQuery(self, *fields, **kwargs)

    def __insert__(self, record, **kwargs):
        version = self.version
        current_key = self.key(record, version)
        next_key = self.key(record, version + 1)
        score = self.score(record)
        dump = record.dump()
        val = str([str(dump[f]) for f in self.schema.fields])
        self.instance.zadd(current_key, score, val)
        self.instance.zadd(next_key, score, val)
        if self.retention is not None:
            self.instance.expireat(current_key, (version + 1) * self.retention)
            self.instance.expireat(next_key, (version + 2) * self.retention)

    def __append__(self, frame, **kwargs):
        records = frame.to_records()
        n = int(self.instance.connection_pool.max_connections / 3)
        with ThreadPoolExecutor(n) as executor:
            executor.map(self.__insert__, records)

    # Redefinition of DataTable.insert to gain speed
    def insert(self, *records, **kwargs):
        """Inserts one or more records into table."""
        n = int(self.instance.connection_pool.max_connections / 3)
        with ThreadPoolExecutor(n) as executor:
            executor.map(self.__insert__, records)

    def maxrows(self, start, end):
        """Estimates max. number of rows for single non-sequential key.

        This requires the schema to implement a rowcount function.
        """
        if self.maxrate is None:
            return NotImplemented
        return self.schema.rowcount(start, end, self.maxrate)

    def __repr__(self):
        string = 'RedisTable[name={nm}](instance={ins})'
        return string.format(ins=self.instance, nm=self.name)


class RedisQueryException(DataQueryException):
    pass


class RedisMaxRowsException(RedisQueryException):
    """Raised when queries reach specified maxrows."""
    pass


class RedisQuery(DataQuery):
    __maxrows__ = 1e5  # Default limit for all queries

    def __init__(self, *fields, **quargs):
        super().__init__(*fields, **quargs)
        if not all(nskey in self.quargs for nskey in self.table.schema.nskeys):
            msg = "Target values for all non-sequential key fields must \
                   be specified in a Redis Query."
            raise RedisQueryException(msg)

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

        Note that queries on sequential keys are closed on the left side
        and open on the right side.
        Params:
            maxrows: limits the number of rows. If None and the table has
                a specified maxrate and sequential key, maxrows is computed
                automatically. An absolute limit of __maxrows__ is set by
                default.
            maxraise: raises if maxrows is exceeded.
        """
        maxrows = fun.get(maxrows, self._maxrows)
        row_count = 0
        if self.seqkeyrange is None:
            min_score = '-inf'
            max_score = '+inf'
        else:
            scoring = self.table.scoring
            min_score, max_score = sorted(scoring(b) for b in self.seqkeyrange)
            if self.table.reverse:
                excluded = min_score
            else:
                excluded = max_score
        version = str(self.table.version)
        if limit is None:
            rd_start, rd_num = None, None
        else:
            rd_start, rd_num = 0, limit
        for g in self.nskeygroups:
            if row_count < maxrows:
                fields = '#'.join(str(k) for k in g)
                key = '#'.join((self.table.name, fields, version))
                results = self.table.instance.zrangebyscore(key,
                                                            min_score,
                                                            max_score,
                                                            rd_start,
                                                            rd_num,
                                                            withscores=True)
                for r, score in results:
                    if score == excluded:
                        continue
                    dump = r.decode()
                    values = [s.strip("'") for s in dump[1:-1].split(', ')]
                    yield tuple(v for i, v in zip(self.fieldmap, values) if i)
                    row_count += 1
                    if row_count >= maxrows:
                        if maxraise:
                            msg = "Query exceeds maxrows {}".format(maxrows)
                            raise RedisMaxRowsException(msg)
                        break
