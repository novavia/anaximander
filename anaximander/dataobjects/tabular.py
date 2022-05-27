from typing import Iterator, Optional

import pandas as pd

from anaximander.descriptors.fields import DataField

from ..utilities import functions as fun
from ..descriptors.datatypes import nxmodel
from ..descriptors.schema import IndexSchema, DataSchema
from ..descriptors.scope import DataScope
from ..descriptors import metadata, metamethod, metaproperty
from ..meta import archetype, trait

from .base import DataObject
from .data import Data
from .columnar import Column
from .models import Record, Fact, Sample, Event, Transition, Session, Journal


@archetype
class Table(DataObject):
    modeltype: type[nxmodel] = metadata(typespec=True)
    datascope: Optional[DataScope] = metadata(objectspec=True)

    def __init__(
        self,
        data=None,
        *,
        parse: bool = True,
        conform: bool = True,
        validate: bool = True,
        integrate: bool = True,
        datascope=None,
        **kwargs,
    ):
        if datascope is not None:
            datascope = DataScope(datascope, tz=self.dataschema.index.tz)
        super().__init__(
            data,
            parse=parse,
            conform=conform,
            validate=validate,
            integrate=integrate,
            datascope=datascope,
            **kwargs,
        )

    @metaproperty("timestamped")
    def has_timestamped_schema(modeltype: type[nxmodel]) -> bool:
        return "timestamped" in modeltype.metacharacters

    @metaproperty("timespanned")
    def has_timespanned_schema(modeltype: type[nxmodel]) -> bool:
        return "timespanned" in modeltype.metacharacters

    @metaproperty("timeblocked")
    def has_timeblocked_schema(modeltype: type[nxmodel]) -> bool:
        return "timeblocked" in modeltype.metacharacters

    @metaproperty("fact")
    def has_fact_modeltype(modeltype: type[nxmodel]) -> bool:
        return issubclass(modeltype, Fact)

    @metaproperty("sample")
    def has_sample_modeltype(modeltype: type[nxmodel]) -> bool:
        return issubclass(modeltype, Sample)

    @metaproperty("event")
    def has_event_modeltype(modeltype: type[nxmodel]) -> bool:
        return issubclass(modeltype, Event)

    @metaproperty("transition")
    def has_transition_modeltype(modeltype: type[nxmodel]) -> bool:
        return issubclass(modeltype, Transition)

    @metaproperty("session")
    def has_session_modeltype(modeltype: type[nxmodel]) -> bool:
        return issubclass(modeltype, Session)

    @metaproperty("journal")
    def has_journal_modeltype(modeltype: type[nxmodel]) -> bool:
        return issubclass(modeltype, Journal)

    @classmethod
    def __init_kwargs__(cls) -> set[str]:
        """Makes admissible keyword arguments for instances."""
        base_kwargs = getattr(cls, "__kwargs__", set())
        try:
            dataschema: DataSchema = getattr(getattr(cls, "modeltype"), "dataschema")
        except AttributeError:
            dataschema = None

        def index_field_retriever(field: DataField):
            dataspec = field.dataspec()
            archetype = getattr(field.type_, "archetype", Data)
            metadata = dataspec.extensions.get("metadata", {})
            fieldtype = archetype.subtype(dataspec=dataspec, **metadata)
            coltype = Column[fieldtype]

            def retriever(table: Table):
                series: pd.Series = getattr(
                    getattr(table, "_data").reset_index(), field.name
                )
                return coltype(
                    series,
                    parse=False,
                    conform=False,
                    validate=False,
                    integrate=False,
                    datascope=table.datascope,
                )

            return retriever

        def payload_field_retriever(field: DataField):
            dataspec = field.dataspec()
            archetype = getattr(field.type_, "archetype", Data)
            metadata = dataspec.extensions.get("metadata", {})
            fieldtype = archetype.subtype(dataspec=dataspec, **metadata)
            coltype = Column[fieldtype]

            def retriever(table: Table) -> pd.Series:
                series: pd.Series = getattr(getattr(table, "_data"), field.name)
                return coltype(
                    series,
                    parse=False,
                    conform=False,
                    validate=False,
                    integrate=False,
                    index_schema=dataschema.index,
                    datascope=table.datascope,
                )

            return retriever

        if isinstance(dataschema, DataSchema):
            for field in dataschema.fields.values():
                if field.name in dataschema.index.fields:
                    retriever = index_field_retriever(field)
                else:
                    retriever = payload_field_retriever(field)
                field.set_attribute(cls, retriever=retriever)
            return base_kwargs | set(dataschema.fields)
        else:
            return base_kwargs

    @metaproperty
    def dataschema(modeltype) -> DataSchema:
        return getattr(modeltype, "dataschema", DataSchema.from_modelspec(modeltype))

    @metaproperty
    def strict_schema(modeltype) -> bool:
        return getattr(modeltype, "strict_schema", True)

    @property
    def index_schema(self) -> IndexSchema:
        return self.dataschema.index

    @metamethod
    def preprocessor(cls, modeltype: type[nxmodel], **metadata):
        """Method invoked by __init__ to resolve data inputs."""

        @classmethod
        def preprocess(cls_, data, **kwargs):
            if isinstance(data, pd.DataFrame):
                return data
            try:
                return pd.DataFrame(data)
            except (ValueError, TypeError):
                msg = f"{cls_} could not interpret {data}"
                raise TypeError(msg)

        return preprocess

    # The following methods are obviously very suboptimal. They will be
    # replaced by an implementation based on Pandera.
    @metamethod
    def parser(cls, modeltype: type[nxmodel], **metadata):
        dataschema: DataSchema = getattr(
            modeltype, "dataschema", DataSchema.from_modelspec(modeltype)
        )
        model_cls = dataschema.model_class(
            namespace=modeltype, parse=True, validate=False
        )
        strict_schema = getattr(modeltype, "strict_schema", True)
        schema_fields = dataschema.fields

        @classmethod
        def parse_strictly(cls_, data: pd.DataFrame):
            return data.reset_index().apply(
                lambda row: model_cls(**row).dict(), axis=1, result_type="expand"
            )

        @classmethod
        def parse_flexibly(cls_, data: pd.DataFrame):
            extra_columns = [c for c in data if c not in schema_fields]
            parsed = data.reset_index().apply(
                lambda row: model_cls(**row).dict(), axis=1, result_type="expand"
            )
            extra_data = data[extra_columns]
            return pd.concat([parsed, extra_data], axis=1)

        if strict_schema:
            return parse_strictly
        else:
            return parse_flexibly

    @metamethod
    def conformer(cls, modeltype: type[nxmodel], **metadata):
        dataschema: DataSchema = getattr(
            modeltype, "dataschema", DataSchema.from_modelspec(modeltype)
        )
        strict_schema = getattr(modeltype, "strict_schema", True)
        schema_fields = dataschema.fields
        schema_dtypes = {name: f.dtype for name, f in schema_fields.items()}

        @classmethod
        def conform(
            cls_, data: pd.DataFrame, datascope: Optional[DataScope] = None, **metadata
        ):
            if data.index.names[0] is not None:
                data = data.reset_index()
            # In the case of a timespan, columns are converted back to start and end times
            if dataschema.is_timespanned:
                if "timespan" in data:
                    data.drop("timespan", axis=1, inplace=True)
            # Gather differences between schema fields and data columns
            schema_columns = set(data) & set(schema_fields)
            missing_columns = set(schema_fields) - schema_columns
            extra_columns = set(data) - schema_columns

            # Raise error if there are missing mandatory fields
            missing_fields = [
                f
                for col in missing_columns
                if not (f := schema_fields[col]).dispensable
            ]
            if missing_fields:
                missing_fields = fun.sort_by(missing_fields, schema_fields)
                msg = f"{data} is missing required columns {[f.name for f in missing_fields]}"
                raise ValueError(msg)

            # Otherwise reorder columns per schema spec
            columns = list(fun.sort_by(schema_columns, schema_fields))

            # Make categories for categorical indexes
            nonlocal schema_dtypes
            schema_dtypes = schema_dtypes.copy()
            if keys := dataschema.index.nxkeys:
                if datascope is None:
                    for key in keys:
                        schema_dtypes[key] = "category"
                else:
                    key_categories = dict()
                    if len(keys) == 1:
                        key_categories[keys[0]] = datascope["keys"]
                    else:
                        key_categories = datascope["keys"] or dict()
                    for key in keys:
                        categories = key_categories.get(key, None)
                        if isinstance(categories, tuple):
                            schema_dtypes[key] = pd.CategoricalDtype(
                                categories=categories, ordered=True
                            )
                        else:
                            schema_dtypes[key] = pd.CategoricalDtype(
                                categories=categories
                            )

            # Checks dtypes
            dtypes = pd.Series({c: schema_dtypes[c] for c in columns})
            try:
                assert dtypes.equals(data[columns].dtypes)
            except AssertionError:
                try:
                    data[columns] = data[columns].astype(dtypes)
                except (ValueError, TypeError):
                    msg = f"Could not conform {data} to specification {dtypes}"
                    raise TypeError(msg)

            # And add extra columns in their original order if the schema is not strict
            if not strict_schema:
                columns += list(fun.sort_by(extra_columns, data))

            conformed = data[columns]
            index_fields = list(dataschema.index.fields)
            # In the case of a timespan, the start and end time are turned
            # into an interval
            if dataschema.is_timespanned:
                span_fields = list(dataschema.index.temporal_fields)
                timespan = lambda row: pd.Interval(row[0], row[1], "left")
                conformed["timespan"] = conformed[span_fields].apply(timespan, axis=1)
                for f in span_fields:
                    index_fields.remove(f)
                index_fields.append("timespan")
            if index_fields:
                return conformed.set_index(index_fields).sort_index()
            else:
                return conformed

        return conform

    @metamethod
    def validator(cls, modeltype: type[nxmodel], **metadata):
        dataschema: DataSchema = getattr(
            modeltype, "dataschema", DataSchema.from_modelspec(modeltype)
        )
        model_cls = dataschema.model_class(
            namespace=modeltype, parse=False, validate=True
        )

        def validate(self):
            data: pd.DataFrame = self._data
            data.reset_index().apply(lambda row: model_cls(**row).dict(), axis=1)
            datascope: Optional[DataScope] = self.datascope
            if datascope is not None:
                # This validates that the dataframe's categories match the scope
                # If they do, then the correponding columns cannot contain
                # values outside of the categories
                if (keys := dataschema.index.nxkeys) and datascope["keys"]:
                    key_categories = dict()
                    if len(keys) == 1:
                        key_categories[keys[0]] = datascope["keys"]
                    else:
                        key_categories = datascope["keys"] or dict()
                    for key in keys:
                        try:
                            assert set(data[key].dtype.categories) == set(
                                key_categories.get(key)
                            )
                        except (TypeError, KeyError):
                            pass
                        except AssertionError:
                            msg = "Key categories don't match the supplied datascope"
                            raise ValueError(msg)
                if time_scope := datascope["time"]:
                    if dataschema.is_timestamped:
                        assert (
                            pd.Series(self.timestamps).apply(lambda t: t in time_scope)
                        ).all()
                    elif dataschema.is_timespanned or dataschema.is_timeblocked:
                        assert (self.timespans.overlaps(time_scope)).all()
            return True

        return validate

    def _row_from_iloc(self, position: int) -> Record:
        value = self._data.reset_index().iloc[position]
        return self.modeltype(
            value, parse=False, conform=True, validate=False, integrate=False
        )

    def _slice_from_iloc(self, *slice_args) -> "Table":
        slice_ = slice(*slice_args)
        data = self._data.iloc[slice_]
        return type(self)(
            data, parse=False, conform=False, validate=False, integrate=False
        )

    def records(self) -> Iterator[nxmodel]:
        """Returns an iterator of records making up self."""
        return (self._row_from_iloc(i) for i in range(self._data.size))


@trait("timestamped")
class TimestampedTable(Table):
    @property
    def timestamps(self) -> pd.DatetimeIndex:
        ts_field = self.dataschema.index.nxtime
        return self._data.index.get_level_values(ts_field)


@trait("timespanned")
class TimespannedTable(Table):
    @property
    def timespans(self) -> pd.IntervalIndex:
        return self._data.index.get_level_values("timespan")

    @property
    def timespan(self) -> pd.IntervalIndex:
        return self._data.index.get_level_values("timespan")

    @property
    def start_times(self) -> pd.DatetimeIndex:
        start_field = self.dataschema.index.nxstart
        return pd.DatetimeIndex(self._data[start_field])

    @property
    def end_times(self) -> pd.DatetimeIndex:
        end_field = self.dataschema.index.nxend
        return pd.DatetimeIndex(self._data[end_field])


@trait("timeblocked")
class TimeblockedTable(Table):
    @property
    def timeblocks(self) -> pd.PeriodIndex:
        tb_field = self.dataschema.index.nxperiod
        return self._data.index.get_level_values(tb_field)

    @property
    def timespans(self) -> pd.IntervalIndex:
        blocks = self.timeblocks
        start_times, end_times = blocks.start_time, blocks.end_time
        spans = pd.DataFrame({"start_time": start_times, "end_time": end_times})
        timespan = lambda row: pd.Interval(row[0], row[1], "left")
        return pd.IntervalIndex(spans.apply(timespan, axis=1))
