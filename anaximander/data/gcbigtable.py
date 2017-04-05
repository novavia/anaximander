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
from itertools import product

from google.cloud.bigtable.instance import Instance
from google.cloud.bigtable.row_filters import ColumnQualifierRegexFilter, \
    RowFilterChain, RowFilterUnion

from ..utilities import functions as fun, nxattr, xprops
from ..meta import prototype, metacharacter
from .schema import Schema
from .annotations import interval
from .table import DataTable, DataQuery, DataQueryException

__all__ = ['BigTableDataTable']

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
                  'timestamp': lambda x: str(int(1e6 * x.timestamp()))}
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


@prototype
@nxattr.s
class BigTableDataTable(DataTable):
    schema = metacharacter(validate=fun.subcheck(Schema))
    instance = nxattr.ib(validator=nxattr.validators.instance_of(Instance))
    name = nxattr.ib(validator=nxattr.validators.instance_of(str))

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

    @xprops.cachedproperty
    def columns(self):
        """A dictionary of the form {family:[column names]}."""
        return btcolumns(*self.schema.fields.values())

    @xprops.cachedproperty
    def rowkey(self):
        """Function that turns a record's key attributes into a row key."""
        return keymaker(self.schema)        

    def __create__(self):
        rval = self.table.create()
        for family in self.columns:
            column_family = self.table.column_family(family)
            column_family.create()
        return rval

    def __remove__(self):
        return self.table.delete()

    def __query__(self, *fields, **kwargs):
        return BigTableQuery(self, *fields, **kwargs)

    def __insert__(self, record, **kwargs):
        rowkey = self.rowkey(*record.keys)
        row = self.table.row(rowkey)
        for family, columns in self.columns.items():
            for col in columns:
                row.set_cell(family,
                             col.encode('utf-8'),
                             str(getattr(record, col)).encode('utf-8'))
        row.commit()

    def __append__(self, frame, **kwargs):
        for record in frame.to_records():
            self.__insert__(record, **kwargs)


class BigTableQueryException(DataQueryException):
    pass


class BigTableQuery(DataQuery):

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
        return (rowkey(*startkeys), rowkey(*endkeys))

    def _make_rowkeypairs(self):
        """Returns an iterable of rowkey pairs to slice the table."""
        nskeys = self.table.schema.nskeys
        seqkey = self.table.schema.seqkey
        seqkeyfield = self.table.schema.keys.get(seqkey, None)
        nskeyquargs = OrderedDict(((k, self.quargs[k]) for k in nskeys))
        groups = product(*nskeyquargs.values())
        if seqkeyfield is not None:
            try:
                seqkeyrange = self.quargs[seqkey]
            except KeyError:
                seqkeyrange = interval(ref=seqkeyfield)
        else:
            seqkeyrange = None
        return (self._rowkeypair(*g, seqkeyrange=seqkeyrange) for g in groups)

    def read_row(self, row):
        attrs = ChainMap(*[{k.decode('utf-8'): v[0].value.decode('utf-8')
                            for k, v in row.cells[family].items()}
                           for family in self.columns])
        return tuple(attrs[k] for k in self.fields)
    
    def __fetch__(self):
        """Fetch primitive."""
        rowkeypairs = self._make_rowkeypairs()
        row_groups = [self.table.table.read_rows(s, e, filter_=self.rowfilter)
                      for s, e in rowkeypairs]
        for g in row_groups:
            g.consume_all()
            for row in g.rows.values():
                yield self.read_row(row)
