#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a base Frame class.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import abc
from collections import OrderedDict
from collections.abc import Mapping, Sequence, Iterator

import numpy as np
import pandas as pd

from ..utilities import functions as fun
from ..utilities import nxattr, xprops
from ..utilities.nxtime import tz_aware
from ..utilities.nxrange import levels, Level, Levels
from ..meta import NxObject, metacharacter, typeinitmethod, newtypemethod, \
    cachedtypeproperty, prototype, clade
from .exceptions import DataError
from .base import IndexedDataObject
from .schema import Schema
from .fields import Field, Raw, Nested, Dict, List, String, UUID, \
    Number, Integer, Decimal, Boolean, FormattedString, Float, DateTime, \
    LocalDateTime, Time, Date, TimeDelta, Url, URL, Email, Method, Function, \
    Str, Bool, Int, Constant, NxDataField, Scalar, Timestamp, Duration, Period
from .annotations import interval, domain
from .record import NxRecord
from .series import NxSeries
from .plot import plot_series

__all__ = ['NxDataFrame', 'NxDataCollection', 'NxDataMapping',
           'NxDataSequence', 'CsvLoader', 'CsvDumper', 'ConformityError']

# =============================================================================
# Field mapping from Schemas to pandas / numpy
# =============================================================================


_field_map = {Field: np.dtype('object'),
              Raw: np.dtype('object'),
              Nested: np.dtype('object'),
              Dict: NotImplemented,
              List: NotImplemented,
              String: np.dtype('object'),
              UUID: np.dtype('object'),
              Number: np.dtype('float'),
              Integer: np.dtype('int'),
              Decimal: np.dtype('float'),
              Boolean: np.dtype('bool'),
              FormattedString: np.dtype('object'),
              Float: np.dtype('float'),
              DateTime: np.dtype('datetime64[ns]'),
              LocalDateTime: np.dtype('datetime64[ns]'),
              Time: np.dtype('datetime64[ns]'),
              Date: np.dtype('datetime64[ns]'),
              TimeDelta: np.dtype('timedelta64[ns]'),
              Url: np.dtype('object'),
              URL: np.dtype('object'),
              Email: np.dtype('object'),
              Method: NotImplemented,
              Function: NotImplemented,
              Str: np.dtype('object'),
              Bool: np.dtype('bool'),
              Int: np.dtype('int'),
              Constant: NotImplemented,
              Timestamp: np.dtype('datetime64[ns]'),
              Duration: np.dtype('timedelta64[ns]'),
              Period: np.dtype('datetime64[ns]')
              }


def dtype(field):
    """Returns a BiqQuery field type from a Schema field or NotImpemented."""
    ftype = type(field)
    if issubclass(ftype, Scalar):
        return field.datatype.dtype
    for ft in ftype.__mro__:
        try:
            return _field_map[ft]
        except KeyError:
            pass
    raise TypeError("Non-field type {0} passed to dtype.".format(field))


def dtypes(schema):
    """Turns a schema class or instance into an OrderedDict of dtypes."""
    return OrderedDict([(k, dtype(v)) for k, v in schema.fields.items()])


def rqdtypes(schema):
    """Reurns an OrderedDict of dtypes for required fieldd only."""
    return OrderedDict([(k, dtype(v)) for k, v in schema.required.items()])


class FrameError(DataError):
    """Specialized exception for Frames."""
    pass


class ConformityError(FrameError):
    """Raised at Frame construction if supplied data doesn't conform."""
    pass

# =============================================================================
# Frame base class
# =============================================================================


class NxDataFrame(IndexedDataObject):
    """A read-only data table with strong column semantic enforcement."""
    schema = None  # placeholder for schema in concrete classes

    @cachedtypeproperty
    def kcols(cls):
        """Returns non-sequential schema key field names."""
        return list(cls.schema.nskeys)

    @cachedtypeproperty
    def sqcol(cls):
        """Returns sequential key field name, if any."""
        return cls.schema.seqkey

    @cachedtypeproperty
    def rowtype(cls):
        return NxRecord[cls.schema]

    @cachedtypeproperty
    def domain(cls):
        """The sequential domain for self, if any."""
        if not cls.sqcol:
            return None
        else:
            return domain(cls.schema.fields[cls.sqcol])

    @cachedtypeproperty
    def frametype(cls):
        """Returns the default NxDataFrame subtype for arbitrary data sets.

        If cls's schema contains non-sequential keys, it is a Mapping.
        Else if there is a sequential key it is a Sequence.
        Otherwise it is a Collection.
        """
        if cls.schema.nskeys:
            return NxDataMapping[cls.schema]
        elif cls.schema.seqkey:
            return NxDataSequence[cls.schema]
        else:
            return NxDataCollection[cls.schema]

    def __init__(self, data=None, context=None, validate=False,
                 ix_range=None, **rgargs):
        """Data can be any admissible data argument to a dataframe.

        params:
            data: a DataFrame of data argument to a DataFrame.
            context: Optional context to populate DataObject's context
                weak property.
            validate: if True, the entire data gets validated against the
                class' schema. Must be used intentionally as there is a
                performance penalty.
            ix_range: an optional range argument for subsetting based on
                self's index. If the index name is also specified in **rgargs,
                a ValueError is raised.
            **rgargs: range arguments. Admissible keys are the names of
                the schema's key fields. Sequential keys accept values that
                can be coerced into a nxrange.Interval, whereas non-sequential
                keys accept values that can be coerced into an
                nxrange.DiscreteRange.

        If validate is False, the method nonetheless performs a schema-level
        validation to verify conformity between the DataFrame's column names
        and types and the class' Schema (see cast method).
        """
        if isinstance(data, NxDataFrame):
            context = context or data.context
            data = data.data
        if ix_range is not None:
            if self.sqcol in rgargs:
                msg = "Specify either the index range or a range for the \
                       sequential key but not both."
                raise ValueError(msg)
            else:
                rgargs[self.sqcol] = ix_range
        self._data = self.cast(data)
        self._subset(**rgargs)
        if validate:
            self.validate()
        self.context = context

    @classmethod
    def cast(cls, data):
        """Casts supplied dataframe-like object to the class' schema.

        data must be a valid input to pandas.DataFrame.
        The method performs the following functions:
        * raises PandasError if data cannot be cast to a DataFrame
        * raises ConformityError if the data misses required columns;
        on the other hand, no error is raised if other columns are missing,
        and extra columns are simply removed.
        * recasts columns to the dtype specified in the schema if necessary;
        * reorders columns to match the schema if necessary.
        """
        df = pd.DataFrame(data)
        columns = []
        conversions = {}
        if df.empty:
            dtypes_ = rqdtypes(cls.schema)
            columns = list(dtypes_)
            dataframe = pd.DataFrame(columns=columns)
        else:
            df_dtypes = OrderedDict(df.dtypes)
            sc_fields = cls.schema.fields.items()
            sc_dtypes = dtypes(cls.schema).items()
            for (col, field), (_, dtype) in zip(sc_fields, sc_dtypes):
                try:
                    col_type = df_dtypes[col]
                except KeyError:
                    if field.required:
                        msg = "Data is missing required field."
                        raise ConformityError(msg)
                else:
                    columns.append(col)
                    if col_type != dtype:
                        conversions[col] = dtype
                        if dtype is np.dtype('bool'):
                            df[col] = df[col].replace('False', 0)
            dataframe = df[columns]
        if conversions:
            try:
                dataframe = dataframe.astype(conversions)
            except ValueError:
                msg = "Data cannot be cast to required types."
                raise ConformityError(msg)
        dataframe = tz_aware(dataframe)
        if cls.sqcol is not None:
            dataframe.index = dataframe[cls.sqcol]
        else:
            dataframe.reset_index(drop=True, inplace=True)
        return dataframe.sort_index()

    @classmethod
    def _keyrange(cls, **rgargs):
        """Validates and converts range arguments as appropriate."""
        result = {}
        for key, value in rgargs.items():
            is_key_field = False
            try:
                field = getattr(cls.schema, key)
                is_key_field = field.key
            except AttributeError:
                pass
            if not is_key_field:
                msg = "Improper argument {0} passed to {1}"
                raise FrameError(msg.format(key, cls))
            if field.sequential:
                result[key] = interval(*value, ref=field)
            else:
                result[key] = value
        return result

    def _subset(self, **rgargs):
        """Subsets a valid dataframe with supplied range arguments.

        Note that the method is not exposed as it is technically weak:
        it updates the keyrange and hence does not robustly support
        iterative calls -it would need to update the keyrange based on the
        existing keyrange and the new supplied arguments.
        """
        keyrange = self._keyrange(**rgargs)
        self.keyrange = keyrange
        data = self.data
        if data is None:
            return None
        elif data.empty:
            return data
        for key, value in keyrange.items():
            field = getattr(self.schema, key)
            if field.sequential:
                lower, upper = value.bounds
                data = data[(data[key] >= lower) & (data[key] <= upper)]
            else:
                target = levels(value)
                if isinstance(target, Level):
                    data = data[target == data[key]]
                elif isinstance(target, Levels):
                    data = data[data[key].isin(target)]
        self._data = data

    def validate(self):
        """Validates all records in a frame against the schema."""
        records = self.data.astype(str).to_dict(orient='records')
        schema = self.schema(many=True)
        schema.validate(records)

    @property
    def distinct(self):
        """Returns distinct combinations in kcols columns."""
        if not self.kcols:
            return list()
        recs = self._data[self.kcols].drop_duplicates().to_records(index=False)
        return list(tuple(r) for r in recs)

    @property
    def empty(self):
        return self.data.empty

    @typeinitmethod
    def _assign_column_proxies(cls):
        """Sets column proxy properties at type creation."""
        for name, field in cls.schema.fields.items():
            setattr(cls, name, ColumnProxy.propertyfactory(field))

    @newtypemethod
    def rebase(cls):
        """Rebases the type if its Schema inherits from a base Schema."""
        # cls.schema is Schema or directly inherits from it
        if len(cls.schema.base_schemas) <= 1:
            return cls
        base_schema = cls.schema.base_schemas[0]
        prototype = type(cls).__archetype__
        base = prototype[base_schema]
        if issubclass(cls, base):
            return cls
        # If cls doesn't inherit from base, we create a new class that does
        new = base.subtype(name=cls.__name__, schema=cls.schema, overtype=True)
        # If additional descriptors exist in cls, they are ported to new
        xattrs = set(cls.__dict__) - set(new.__dict__)
        for k in xattrs:
            setattr(new, k, getattr(cls, k))
        return new

    def __repr__(self):
        base = '<{arc}[{dt}]({content})>'
        return base.format(arc=clade(self).__name__,
                           dt=self.schema.__name__,
                           content=fun.spformat(self._data, 'row'))

    def __str__(self):
        return self._data.__str__()

    # Plotting

    def plot(self, field=None, ax='new', **kwargs):
        """Customized plot function.

        Params:
            field: a fieldname to plot. Defaults to the first non-key field.
            ax: 'new' for new plot, otherwise see pandas series plot.
        """
        try:
            field = fun.get(field, self.schema.nonkeyfieldnames[0])
        except IndexError:
            msg = "Could not determine which field to plot."
            raise FrameError(msg)
        series = getattr(self, field)()
        ax = plot_series(series, ax, **kwargs)
        return ax

    # I/O methods

    def to_csv(self, filepath_or_buffer, *args, **kwargs):
        """See pandas.to_csv for method signature.

        Note that index is set to False automatically.
        """
        CsvDumper(self, filepath_or_buffer, *args, **kwargs)()

    @classmethod
    def from_csv(cls, filepath_or_buffer, *args, **kwargs):
        """See pandas.from_csv for method signature."""
        loader = CsvLoader(cls.schema, filepath_or_buffer, *args, **kwargs)
        return loader(cls)

    def to_records(self):
        return list(self.iloc)

    # Extension / Contraction methods

    def crop(self, start=None, end=None):
        """Returns a new instance that has been cropped by start and end."""
        sqrange = interval(start, end, ref=self.domain)
        start, end = sqrange
        if self.empty:
            df = self.data
        elif start is not None and end is not None:
            df = pd.DataFrame(self.data.loc[start:end])
        elif end is not None:
            df = pd.DataFrame(self.data.loc[:end])
        elif start is not None:
            df = pd.DataFrame(self.data.loc[start:])
        else:
            df = self.data
        result = type(self)(df, context=self.context)
        if self.lower is not None:
            lower = max((self.lower, start))
        else:
            lower = start
        if self.upper is not None:
            upper = min((self.upper, end))
        else:
            upper = end
        result.keyrange[self.sqcol] = interval(lower, upper, ref=self.domain)
        return result

    @classmethod
    def from_records(cls, records, context=None, **rgargs):
        """Instantiates an NxDataFrame from an iterable of records.

        Records must be instances of NxRecord[self.schema].
        """
        rows = [r.as_dict() for r in records]
        data = pd.DataFrame.from_records(rows)
        return cls(data, context=context, **rgargs)

    def extend(self, data):
        """Extends self with another NxDataFrame or pandas.DataFrame.

        One side effect is the elimination of range metadata, as it becomes
        undefined.
        """
        if isinstance(data, NxDataFrame):
            data = data.data
        if self.empty:
            xdata = data
        else:
            xdata = self._data.append(data)
        self._data = self.cast(xdata)
        self.keyrange = {}

    def append(self, *records):
        """Appends one or more records to self.

        Records must be instances of NxRecord[self.schema].
        """
        rows = [r.as_dict() for r in records]
        data = pd.DataFrame.from_records(rows)
        self.extend(data)


@prototype
class NxDataCollection(NxDataFrame):
    """An NxDataFrame with no indexing capabilities besides row sequence."""
    schema = metacharacter(validate=lambda s: issubclass(s, Schema))


@prototype
class NxDataMapping(NxDataFrame, traits=(Mapping,)):
    """An NxDataFrame indexed by non-sequential key fields in the schema."""
    schema = metacharacter(validate=lambda s: issubclass(s, Schema))

    def __iter__(self):
        if self.kcols:
            return iter(self.distinct)
        else:
            return iter(self.index)

    def __getitem__(self, key):
        """Selects any number of rows and returns appropriate DataObject.

        If self.kcols is an empty list, this method is equivalent to iloc.

        Otherwise the method sccepts the following key arguments:
        * A scalar value, corresponding to the first column in cls.kcols.
        * A list, providing multiple values for the first column in cls.kcols.
        * A tuple, of which the elements are interpreted to match the columns
        in cls.kcols. A None value is a wild card that will not downselect.
        * If a tuple, the tuple can contain lists of values that will be
        matched against the appropriate key column.

        The return value is cast to the appropriate type as follows:
        * If the downselected data features more than one value of combination
        of values in the key columns, an NxDataMapping of the same type is
        returned.
        * If the downselected data features a single value or combination
        of value in the key columns, there are three possibilities:
            * If the schema features a sequential key field, then the
            return value is an NxDataSequence for the same schema.
            * Otherwise, if there are multiple rows, the return value is an
            NxDataCollection for the same schema.
            * If there is no sequential key field in the schema and the
            downselected data features a single row, then an NxRecord is
            returned.

        If no row can be selected, a KeyError is raised.
        """
        if not self.kcols:
            return self.iloc.__getitem__(key)
        if isinstance(key, tuple):
            conditions = []
            for v, c in zip(key, self.kcols):
                if isinstance(v, list):
                    condition = self._data[c].isin(v)
                else:
                    condition = self._data[c] == v
            conditions.append(condition)
            data = self._data[np.logical_and.reduce(conditions)]
        else:
            col = self.kcols[0]
            if isinstance(key, list):
                data = self._data[self._data[col].isin(key)]
            else:
                data = self._data[self._data[col] == key]
        if data.empty:
            raise KeyError()
        if len(data.drop_duplicates(subset=self.kcols)) == 1:
            schema = self.schema
            if self.sqcol is not None:
                return NxDataSequence[schema](data, context=self.context)
            elif len(data) > 1:
                return NxDataCollection[schema](data, context=self.context)
            else:
                return NxRecord[schema](data.ix[0], context=self.context)
        else:
            return type(self)(data, context=self.context)


@prototype
class NxDataSequence(NxDataFrame, traits=(Sequence,)):
    """An NxDataFrame indexed by a unique sequential key field."""
    schema = metacharacter(validate=lambda s: issubclass(s, Schema))

    def __init__(self, data=None, context=None, validate=False,
                 ix_range=None, **rgargs):
        super().__init__(data, context, validate, ix_range, **rgargs)
        self.verify()

    def __getitem__(self, key):
        """Equivalent to self.iloc.__getitem__."""
        return self.iloc.__getitem__(key)

    def verify(self):
        """Checks that the underlying data has a unique key wrt. kcols."""
        if len(self.distinct) > 1:
            msg = "DataSequence {0} features distinct values in key fields."
            raise ConformityError(msg.format(self))

    @property
    def lower(self):
        """Start metadata."""
        try:
            return self.keyrange[self.sqcol].lower
        except (KeyError, IndexError):
            return None

    @property
    def upper(self):
        """End metadata."""
        try:
            return self.keyrange[self.sqcol].upper
        except (KeyError, IndexError):
            return None

# =============================================================================
# Column Proxy class
# =============================================================================


@nxattr.s
class ColumnProxy:
    """A reference to a column in a NxDataFrame.

    NxDataFrames carry properties named after their columns that return
    ColumnProxy objects. The ColumnProxy serves to provide metadata
    information and summarization methods, and upon being called, it will
    return the underlying data, either as an NxSeries object or a regular
    pandas Series.

    Params:
        frame: an NxDataFrame
        field: a schema field
    """
    frame = nxattr.ib(validator=nxattr.validators.instance_of(NxDataFrame))
    field = nxattr.ib()

    @property
    def name(self):
        """Returns the corresponding column name."""
        return self.field.name

    @property
    def column(self):
        """Returns the corresponding column in self.frame.data."""
        return self.frame.data[self.name]

    @property
    def dtype(self):
        """Returns the dtype of the underlying data."""
        return self.column.dtype

    @property
    def ftype(self):
        """Returns the field type."""
        return type(self.field)

    def __call__(self):
        series = self.column
        if isinstance(self.field, NxDataField):
            datatype = self.field.datatype
            context = self.frame.context
            return NxSeries[datatype](series, context=context)
        else:
            return series

    def describe(self):
        return self.column.describe()

    @property
    def min(self):
        return self.column.min()

    @property
    def max(self):
        return self.column.max()

    @property
    def unique(self):
        return set(self.column.unique())

    @classmethod
    def propertyfactory(cls, field):
        """Returns a property based on supplied field."""
        doc = """Returns a ColumnProxy object for column {c}"""
        doc.format(c=field.name)
        return property(lambda self: cls(self, field), doc=doc)

# =============================================================================
# I/O helpers
# =============================================================================


class DataIOException(DataError):
    """Specialized exception for handling I/O exceptions."""
    pass


class DataLoader(NxObject):
    """A representation of a pending data loading operation.

    DataLoader serves as a base class for concrete loading operations such
    as database table queries and fetching data from files.
    DataLoader object only stock references to loading protocol and data
    selection criteria, until their __call__ method is called, which
    tries to execute the loading operation and generate results in the form
    of one or more DataObjects.
    A DataLoader requires a Schema type and store to be instantiated. Beyond
    then, various methods exist to return more precise DataLoaders by
    composition of attributes. This works similarly to, say, SQLAlchemy Query
    object.

    Args:
        schema: A Schema subtype.
        store: any object pointing to a storage resource.
    """

    def __init__(self, schema, store, *args, **kwargs):
        self.schema = schema
        self.store = store
        self.args = args
        self.kwargs = kwargs
        super().__init__()

    @xprops.cachedproperty
    def data(self):
        return None

    @property
    def loaded(self):
        return self.data is not None

    @property
    def frametype(self):
        """The default NxDataFrame subtype for loading data."""
        return NxDataCollection[self.schema].frametype

    @abc.abstractmethod
    def __load__(self):
        """Runs and returns raw results of the underlying loading operation."""
        pass

    def load(self):
        """Executes the query and caches the raw data."""
        self._data = self.__load__()

    def __call__(self, frametype=None):
        """Returns the data in the appropriate NxDataFrame subtype.

        By default, the return value's type is self.channel.frametype.
        However the default can be overriden by passing an explicit frametype.
        The call can also return a generator of NxDataFrames if the __run__
        method returns an Iterator.
        """
        frametype = frametype or self.frametype
        if not self.loaded:
            self.load()
        if isinstance(self.data, Iterator):
            return (frametype(d) for d in self.data)
        else:
            return frametype(self.data)


class DataDumper(NxObject):
    """A representation of a pending data dumping operation.

    This is the counterpart of DataLoader for dumping data to persistent
    storage through a channel.

    Args:
        frame: an NxDataFrame instance
        store: any object pointing to a storage resource.
    """

    def __init__(self, frame, store, *args, **kwargs):
        self.frame = frame
        self.store = store
        self.args = args
        self.kwargs = kwargs
        super().__init__()

    @property
    def schema(self):
        return self.frame.schema

    @xprops.cachedproperty
    def dumped(self):
        return False

    @abc.abstractmethod
    def __dump__(self):
        """Dumps self's data to the designated store."""
        pass

    def dump(self):
        if self.dumped:
            raise DataIOException("Data already dumped.")
        rval = self.__dump__()
        self._dumped = True
        return rval

    def __call__(self):
        return self.dump()


class CsvLoader(DataLoader):
    """Loader for .csv files.

    Reading is based on pd.read_csv and the __call__ method takes all
    the arguments from pd.read_csv. The file may be a path or an
    actual file object.
    """

    def __load__(self):
        return pd.read_csv(self.store, *self.args, **self.kwargs)


class CsvDumper(DataDumper):
    """Dumper for .csv files.

    This is based on pd.DataFrame.to_csv.
    """

    def __init__(self, frame, store, *args, **kwargs):
        super().__init__(frame, store, *args, **kwargs)
        self.kwargs.setdefault('index', False)

    def __dump__(self):
        self.frame.data.to_csv(self.store, *self.args, **self.kwargs)
