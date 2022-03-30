from collections.abc import Mapping
from copy import copy
import dataclasses
from statistics import mode
from typing import Optional, Union
from numpy import isin, result_type

from frozendict import frozendict
from pydantic import BaseModel, create_model
import pandas as pd

from anaximander.descriptors.fields import DataField

from ..utilities.nxdataclasses import DataClass
from ..utilities import functions as fun
from ..descriptors.base import DescriptorRegistry
from ..descriptors.datatypes import datatype, nxdata, nxmodel
from ..descriptors.dataspec import DataSpec
from ..descriptors.schema import DataIndex, DataSchema
from ..descriptors.scope import DataScope
from ..descriptors import (
    metadata,
    option,
    metamethod,
    metaproperty,
    _dataspec_keys,
    _pydantic_validator_keys,
)
from ..meta import Object, archetype, metamorph, trait


NoneType = type(None)


@archetype
class DataObject(Object.Base):
    """Abstract base archetype for all data archetypes."""

    parse: bool = option(True)
    conform: bool = option(True)
    validate: bool = option(True, alias="validate_")
    integrate: bool = option(True)

    def __init__(
        self,
        data=None,
        *,
        parse: bool = True,
        conform: bool = True,
        validate: bool = True,
        integrate: bool = True,
        **kwargs,
    ):
        super().__init__(
            parse=parse,
            conform=conform,
            validate=validate,
            integrate=integrate,
            **kwargs,
        )
        self._data = None
        self._parsed = False
        self._conformed = False
        self._validated = False
        self._integrated = False

        if isinstance(data, DataObject):
            data = data.data

        # TODO: This is brute-force logic that is largely suboptimal in most cases; it
        # will need to be refactored such that data conformity can be assessed
        # first thing, and extra mingling can be bypassed for conforming data
        # A further gotcha is that the implementation works in so far as Pydantic
        # models tolerate extraneous inputs, and hence the kwargs are appended
        # to the data irrespective of what they are. More caution may be
        # required when other parsing / conforming methods are used.

        data = self.preprocessor(data, **kwargs)

        # parsing
        if self.options.parse:
            try:
                data = self.parser(data)
            except Exception as e:
                msg = f"Could not parse {data} into a {repr(type(self))} instance"
                raise TypeError(msg)
            else:
                self._parsed = True

        # conforming
        if self.options.conform:
            try:
                if isinstance(self, (Data, Model)) and self._parsed:
                    pass
                else:
                    data = self.conformer(data, **self.metadata)
            except Exception as e:
                msg = f"Could not conform {data} into a {repr(type(self))} instance"
                raise TypeError(msg)
            else:
                self._conformed = True

        self._data = data

        # data validation
        if self.options.validate:
            try:
                self.validator()
            except Exception as e:
                msg = f"Could not validate {self.data} as a {repr(type(self))} instance"
                raise ValueError(msg)
            else:
                self._validated = True

        # data integration
        if self.options.integrate:
            try:
                self.integrator()
            except Exception as e:
                msg = f"Could not integrate {self}"
                raise ValueError(msg)
            else:
                self._integrated = True

    @property
    def data(self):
        return copy(self._data)

    def __copy__(self):
        copy_ = type(self)(
            self._data,
            parse=False,
            conform=False,
            validate=False,
            integrate=False,
            **self.metadata,
        )
        copy_._options = self._options
        copy_._parsed = self._parsed
        copy_._conformed = self._conformed
        copy_._validated = self._validated
        copy_._integrated = self._integrated
        return copy_

    @metamethod
    def preprocessor(cls, **metadata):
        """Method invoked by __init__ to resolve data inputs."""

        @classmethod
        def preprocess(cls_, data, **kwargs):
            return data

        return preprocess

    @metamethod
    def parser(cls, **metadata):
        @classmethod
        def parse(cls_, data):
            return data

        return parse

    @metamethod
    def conformer(cls, **metadata):
        @classmethod
        def conform(cls_, data, **metadata):
            return NotImplemented

        return conform

    @metamethod
    def validator(cls, **metadata):
        def validate(self):
            return True

        return validate

    @metamethod
    def integrator(cls, **metadata):
        def integrate(self):
            return True

        return integrate

    @property
    def parsed(self):
        """True if the object's data was explicitly parsed at initialization."""
        return self._parsed

    @property
    def parsed(self):
        """True if the object's data was explicitly conformed at initialization."""
        return self._conformed

    @property
    def constructed(self):
        """True if the object's data has been assigned."""
        return self._data is not None

    @property
    def validated(self):
        """True if the object's data was explicitly validated at initialization."""
        return self._validated

    @property
    def integrated(self):
        """True if the object's data was explicitly integrated at initialization."""
        return self._integrated

    @classmethod
    def __data_interpreter__(cls, namespace, *, new_type_name: str) -> dict:
        return {}

    @classmethod
    def __init_kwargs__(cls) -> set[str]:
        """Makes admissible keyword arguments for instances."""
        return getattr(cls, "__kwargs__", set())

    def __eq__(self, other):
        if isinstance(self.data, (pd.Series, pd.DataFrame)):
            data_equality = self.data.equals(other.data)
        else:
            data_equality = self.data == other.data
        return super().__eq__(other) and data_equality

    def __hash__(self):
        data_string = str(self.data)
        metadata_string = str(self.metadata)
        return hash(data_string + metadata_string)


DataType = Union[type[nxdata], type[nxmodel], type[DataObject]]  # type: ignore


@archetype
class Data(DataObject):
    dataspec: DataSpec = metadata(typespec=True)

    @dataspec.parser
    def _convert_dataspec(value):
        if isinstance(value, type):
            if issubclass(value, Data):
                return Data.dataspec
            elif issubclass(value, nxdata):
                return DataSpec(value)
        return value

    @classmethod
    def __data_interpreter__(cls, namespace, *, new_type_name: str) -> dict:
        base_spec: Optional[DataSpec] = getattr(cls, "dataspec", None)
        namespace_spec: Optional[DataSpec] = namespace.get("dataspec")
        if base_spec and namespace_spec:
            if namespace_spec >= base_spec:
                base_spec = namespace_spec
            else:
                raise TypeError
        elif namespace_spec:
            base_spec = namespace_spec
        elif base_spec is None:
            return {}
        base_descriptors = getattr(cls, "__descriptors__")
        descriptor_registry = DescriptorRegistry(parent=base_descriptors)
        descriptor_registry.collect(namespace, "data", strip=True)
        extensions = {k: v for k, v in namespace.items() if k in _dataspec_keys}
        pydantic_validators = {
            k: v for k, v in namespace.items() if k in _pydantic_validator_keys
        }
        dataspec = DataSpec.from_registry(
            descriptor_registry,
            new_type_name=new_type_name,
            base=base_spec,
            **extensions,
            **pydantic_validators,
        )
        return dict(dataspec=dataspec)

    @metamethod
    def preprocessor(cls, dataspec: DataSpec, **metadata):
        """Method invoked by __init__ to resolve data inputs."""

        @classmethod
        def preprocess(cls_, data, **kwargs):
            return data

        return preprocess

    @metamethod
    def parser(cls, dataspec: DataSpec, **metadata):
        model_cls = dataspec.model_class(namespace=cls, parse=True, validate=False)

        @classmethod
        def parse(cls_, data):
            return model_cls(data=data).data

        return parse

    @metamethod
    def conformer(cls, dataspec: DataSpec, **metadata):
        @classmethod
        def conform(cls_, data, **metadata):
            if not isinstance(data, dataspec.pytype):
                msg = f"Incorrect data type {type(data)} supplied to {cls_}"
                raise TypeError(msg)
            return data

        return conform

    @metamethod
    def validator(cls, dataspec: DataSpec, **metadata):
        model_cls = dataspec.model_class(namespace=cls, parse=False, validate=True)

        def validate(self):
            model_cls(data=self._data)
            return True

        return validate

    def __get_validators__(cls):
        """Makes Data types compatible with Pydantic models."""
        yield cls

    def __eq__(self, other: "Data"):
        try:
            data_equality = self.data == other.data
            type_compatibility = (self.dataspec >= other.dataspec) or (
                self.dataspec <= other.dataspec
            )
        except AttributeError:
            return False
        return data_equality and type_compatibility


nxdata.register(Data.basetype)
nxdata.register(Data)


class Integer(Data):
    dataspec = DataSpec(int)


class Float(Data):
    dataspec = DataSpec(float)


@archetype
class Measurement(Float):
    unit: str = metadata()


# For now this archetype serves as a general-purpose series representation
# but with no underlying schema it doesn't specify the indexing scheme at
# the type level. Instead (and pending) it exposes an index as metadata at
# the instance level. The primary use case is to represent columns extracted
# from a Table instance. In that case, the resulting data object inherits the
# index of the Table instance. The gap this leaves is for first-class data Series
# that may be persisted as such on disk, and hence would require an index specifcation.
# This can initially be covered by a single-payload column Table -which would be
# generated by specifying a Record class whose schema has a single non-indexing
# field. The drawback of that approach is that a single-column dataframe is not
# quite the same interface as a series. Hence a more complete treatment would
# involve renaming the present archetype to Column, and offering a separate
# Series archetype that takes a Record type as its data specification, with the
# obvious constraint that the Record's schema features a single payload column.
# Addendum 2-18-22: Another distinction that needs to be made is between a
# data structure with an arbitrary index and one in which the index is obtained
# through a range query -that may be the better way to think about Column vs
# Series.
@archetype
class Column(DataObject):
    datatype: type[nxdata] = metadata(typespec=True)
    dataindex: Optional[DataIndex] = metadata()
    datascope: Optional[DataScope] = metadata()

    @datatype.parser
    def _convert_datatype(value):
        if isinstance(value, DataSpec):
            return Data[value]
        return value

    @metaproperty
    def dataspec(datatype) -> DataSpec:
        return getattr(datatype, "dataspec", DataSpec(datatype))

    @metamethod
    def preprocessor(cls, datatype: type[nxdata], **metadata):
        """Method invoked by __init__ to resolve data inputs."""
        name = getattr(getattr(datatype, "dataspec", None), "name", None)

        @classmethod
        def preprocess(cls_, data, **kwargs):
            if isinstance(data, pd.Series):
                return data
            try:
                series = pd.Series(data)
                series.name = name
                return series
            except (ValueError, TypeError):
                msg = f"{cls_} could not interpret {data}"
                raise TypeError(msg)

        return preprocess

    # The following methods are obviously very suboptimal. They will be
    # replaced by an implementation based on Pandera.
    @metamethod
    def parser(cls, datatype: type[nxdata], **metadata):
        dataspec = getattr(datatype, "dataspec", DataSpec(datatype))
        model_cls = dataspec.model_class(namespace=datatype, parse=True, validate=False)

        @classmethod
        def parse(cls_, data: pd.Series):
            return data.apply(lambda v: getattr(model_cls(data=v), "data"))

        return parse

    @metamethod
    def conformer(
        cls,
        datatype: type[nxdata],
        dataindex: Optional[DataIndex] = None,
        datascope: Optional[DataScope] = None,
        **metadata,
    ):
        dataspec: Optional[DataSpec] = getattr(datatype, "dataspec", None)
        name = getattr(dataspec, "name", None)
        categorical = getattr(dataspec, "extensions", {}).get("nx_key", False)
        if categorical:
            if datascope is None:
                dtype = pd.CategoricalDtype()
            else:
                key_categories = datascope.get("keys", None)
                if isinstance(key_categories, frozendict):
                    categories = key_categories.get(name, None)
                else:
                    categories = key_categories
                if isinstance(categories, tuple):
                    dtype = pd.CategoricalDtype(categories=categories, ordered=True)
                else:
                    dtype = pd.CategoricalDtype(categories=categories)

        @classmethod
        def conform(cls_, data: pd.Series, **metadata):
            data.name = name
            if categorical:
                return data.astype(dtype)
            return data

        return conform

    @metamethod
    def validator(cls, datatype: type[nxdata], **metadata):
        dataspec = getattr(datatype, "dataspec", DataSpec(datatype))
        model_cls = dataspec.model_class(namespace=datatype, parse=False, validate=True)

        def validate(self):
            data: pd.Series = self._data
            data.apply(lambda v: getattr(model_cls(data=v), "data"))
            return True

        return validate

    def __eq__(self, other: "Data"):
        try:
            data_equality = self.data.equals(other.data)
            type_compatibility = (self.dataspec >= other.dataspec) or (
                self.dataspec <= other.dataspec
            )
        except AttributeError:
            return False
        return data_equality and type_compatibility


@archetype
class Model(DataObject):
    """Base class for map-like data structures."""

    dataschema: DataSchema = metadata(typespec=True)
    strict_schema: bool = metadata(True, typespec=True)

    @classmethod
    def __data_interpreter__(cls, namespace, *, new_type_name: str) -> dict:
        base_schema: Optional[DataSchema] = getattr(cls, "dataschema", None)
        base_descriptors = getattr(cls, "__descriptors__")
        descriptor_registry = DescriptorRegistry(parent=base_descriptors)
        descriptor_registry.collect(namespace, "data", strip=True)
        dataschema = DataSchema.from_registry(
            descriptor_registry, new_type_name=new_type_name, base=base_schema
        )
        return dict(dataschema=dataschema)

    @classmethod
    def __init_kwargs__(cls) -> set[str]:
        """Makes admissible keyword arguments for instances."""
        base_kwargs = getattr(cls, "__kwargs__", set())
        schema = getattr(cls, "dataschema", None)

        def field_transformer(field: DataField):
            dataspec = field.dataspec()
            fieldtype = Data[dataspec]

            def transformer(value):
                return fieldtype(
                    value,
                    parse=False,
                    conform=False,
                    validate=False,
                    integrate=False,
                )

            return transformer

        if isinstance(schema, DataSchema):
            for field in schema.fields.values():
                field.set_attribute(cls, transformer=field_transformer(field))
            return base_kwargs | set(schema.fields)
        else:
            return base_kwargs

    @metaproperty("entity")
    def has_entity_schema(dataschema: DataSchema) -> bool:
        return dataschema.is_entity_schema

    @metaproperty("record")
    def has_record_schema(dataschema: DataSchema) -> bool:
        return dataschema.is_record_schema

    @metaproperty("spec")
    def has_spec_schema(dataschema: DataSchema) -> bool:
        return dataschema.is_spec_schema

    @metaproperty("tabular")
    def has_tabular_schema(dataschema: DataSchema) -> bool:
        return dataschema.is_tabular

    @metaproperty("nested")
    def has_nested_schema(dataschema: DataSchema) -> bool:
        return dataschema.is_nested

    @metamethod
    def preprocessor(cls, dataschema: DataSchema, **metadata):
        """Method invoked by __init__ to resolve data inputs."""

        @classmethod
        def preprocess(cls_, data, **kwargs) -> dict:
            if not any(name in dataschema.fields for name in kwargs):
                if isinstance(data, (BaseModel, pd.Series, Mapping)):
                    return data
                elif isinstance(data, DataClass):
                    return dataclasses.asdict(data)
                else:
                    msg = f"{cls_} could not interpret {data}"
                    raise TypeError(msg)
            else:
                for key, value in kwargs.copy().items():
                    if isinstance(value, DataObject):
                        kwargs[key] = value._data
                if isinstance(data, (BaseModel, pd.Series, Mapping)):
                    data = dict(data) | kwargs
                if isinstance(data, DataClass):
                    data = dataclasses.asdict(data) | kwargs
                if data is None:
                    data = kwargs
                return data

        return preprocess

    @metamethod
    def parser(cls, dataschema: DataSchema, strict_schema: bool, **metadata):
        model_cls = dataschema.model_class(namespace=cls, parse=True, validate=False)

        @classmethod
        def parse_strictly(cls_, data: Union[BaseModel, pd.Series, Mapping]):
            return model_cls.parse_obj(data)

        @classmethod
        def parse_flexibly(cls_, data: Union[BaseModel, pd.Series, Mapping]):
            xdata = {k: v for k, v in dict(data).items() if k not in dataschema.fields}
            xfields = [DataField.from_key_value_pair(k, v) for k, v in xdata.items()]
            xschema = dataschema.extend(fields=xfields)
            xmodel_cls = xschema.model_class(namespace=cls, parse=True, validate=False)
            return xmodel_cls.parse_obj(data)

        if strict_schema:
            return parse_strictly
        else:
            return parse_flexibly

    @metamethod
    def conformer(cls, dataschema: DataSchema, strict_schema: bool, **metadata):
        model_cls = dataschema.model_class(namespace=cls, parse=False, validate=False)

        @classmethod
        def conform_strictly(
            cls_, data: Union[BaseModel, pd.Series, Mapping], **metadata
        ):
            return model_cls.parse_obj(data)

        @classmethod
        def conform_flexibly(
            cls_, data: Union[BaseModel, pd.Series, Mapping], **metadata
        ):
            xdata = {k: v for k, v in dict(data).items() if k not in dataschema.fields}
            xfields = [DataField.from_key_value_pair(k, v) for k, v in xdata.items()]
            xschema = dataschema.extend(fields=xfields)
            xmodel_cls = xschema.model_class(namespace=cls, parse=False, validate=False)
            return xmodel_cls.parse_obj(data)

        if strict_schema:
            return conform_strictly
        else:
            return conform_flexibly

    @metamethod
    def validator(cls, dataschema: DataSchema, strict_schema: bool, **metadata):
        model_cls = dataschema.model_class(namespace=cls, parse=False, validate=True)

        def validate_strictly(self):
            data: Union[BaseModel, pd.Series] = self._data
            model_cls.parse_obj(data)
            return True

        def validate_flexibly(self):
            data: Union[BaseModel, pd.Series] = self._data
            xdata = {k: v for k, v in dict(data).items() if k not in dataschema.fields}
            xfields = [DataField.from_key_value_pair(k, v) for k, v in xdata.items()]
            xschema = dataschema.extend(fields=xfields)
            xmodel_cls = xschema.model_class(namespace=cls, parse=False, validate=True)
            xmodel_cls.parse_obj(data)
            return True

        if strict_schema:
            return validate_strictly
        else:
            return validate_flexibly


@trait("tabular")
class TabularModel(Model):
    pass


@trait("nested")
class NestedModel(Model):
    pass


@metamorph
@archetype
class Entity(Model, metacharacters=["entity"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def is_entity_schema(schema: DataSchema):
        """Schema must be an entity schema"""
        return schema.is_entity_schema


@metamorph
@archetype
class Record(Model, metacharacters=["record"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def is_record_schema(schema: DataSchema):
        """Schema must be a record schema"""
        return schema.is_record_schema

    @metamethod
    def parser(cls, dataschema: DataSchema, strict_schema: bool, **metadata):
        model_cls = dataschema.model_class(namespace=cls, parse=True, validate=False)

        @classmethod
        def parse_strictly(cls_, data: Union[BaseModel, pd.Series, Mapping]):
            return pd.Series(model_cls.parse_obj(data).dict())

        @classmethod
        def parse_flexibly(cls_, data: Union[BaseModel, pd.Series, Mapping]):
            xdata = {k: v for k, v in dict(data).items() if k not in dataschema.fields}
            xfields = [DataField.from_key_value_pair(k, v) for k, v in xdata.items()]
            xschema = dataschema.extend(fields=xfields)
            xmodel_cls = xschema.model_class(namespace=cls, parse=True, validate=False)
            return pd.Series(xmodel_cls.parse_obj(data).dict())

        if strict_schema:
            return parse_strictly
        else:
            return parse_flexibly

    @metamethod
    def conformer(cls, dataschema: DataSchema, strict_schema: bool, **metadata):
        model_cls = dataschema.model_class(namespace=cls, parse=False, validate=False)

        @classmethod
        def conform_strictly(
            cls_, data: Union[BaseModel, pd.Series, Mapping], **metadata
        ):
            return pd.Series(model_cls.parse_obj(data).dict())

        @classmethod
        def conform_flexibly(
            cls_, data: Union[BaseModel, pd.Series, Mapping], **metadata
        ):
            xdata = {k: v for k, v in dict(data).items() if k not in dataschema.fields}
            xfields = [DataField.from_key_value_pair(k, v) for k, v in xdata.items()]
            xschema = dataschema.extend(fields=xfields)
            xmodel_cls = xschema.model_class(namespace=cls, parse=False, validate=False)
            return pd.Series(xmodel_cls.parse_obj(data).dict())

        if strict_schema:
            return conform_strictly
        else:
            return conform_flexibly

    @metaproperty("keyed")
    def has_keyed_schema(dataschema: DataSchema) -> bool:
        return dataschema.is_keyed

    @metaproperty("atemporal")
    def has_atemporal_schema(dataschema: DataSchema) -> bool:
        return not dataschema.is_temporal

    @metaproperty("timestamped")
    def has_timestamped_schema(dataschema: DataSchema) -> bool:
        return not dataschema.is_timestamped

    @metaproperty("timespanned")
    def has_timespanned_schema(dataschema: DataSchema) -> bool:
        return not dataschema.is_timespanned

    @metaproperty("timeblocked")
    def has_timeblocked_schema(dataschema: DataSchema) -> bool:
        return not dataschema.is_timeblocked


@metamorph
@archetype
class Fact(Record, metacharacters=["atemporal"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def is_atemporal_schema(schema: DataSchema):
        """Schema must be atemporal"""
        return not schema.is_temporal


@archetype
class Sample(Record, metacharacters=["timestamped"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def is_timestamped_schema(schema: DataSchema):
        """Schema must feature a timestamp"""
        return schema.is_timestamped


@archetype
class Event(Record, metacharacters=["timestamped"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def is_timestamped_schema(schema: DataSchema):
        """Schema must feature a timestamp"""
        return schema.is_timestamped


@archetype
class Transition(Event):
    pass


@archetype
class Session(Record, metacharacters=["timespanned"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def is_timespanned_schema(schema: DataSchema):
        """Schema must feature start and end times"""
        return schema.is_timespanned

    @property
    def timespan(self):
        start_field, end_field = self.dataschema.index.temporal_fields
        start_time = pd.to_datetime(self._data[start_field])
        end_time = pd.to_datetime(self._data[end_field])
        return pd.Interval(start_time, end_time, "left")


@archetype
class Journal(Record, metacharacters=["timeblocked"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def is_timestamped_schema(schema: DataSchema):
        """Schema must feature a period"""
        return schema.is_timeblocked


@metamorph
@archetype
class Spec(Model, metacharacters=["spec"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def is_spec_schema(schema: DataSchema):
        """Schema cannot feature an index"""
        return schema.is_spec_schema


@archetype
class Table(DataObject):
    modeltype: type[nxmodel] = metadata(typespec=True)
    datascope: Optional[DataScope] = metadata(objectspec=True)

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
            fieldtype = Data[dataspec]
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
            fieldtype = Data[dataspec]
            coltype = Column[fieldtype]

            def retriever(table: Table) -> pd.Series:
                series: pd.Series = getattr(getattr(table, "_data"), field.name)
                return coltype(
                    series,
                    parse=False,
                    conform=False,
                    validate=False,
                    integrate=False,
                    dataindex=dataschema.index,
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
                return conformed.set_index(index_fields)
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
                if keys := dataschema.index.nxkeys:
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
                            return False
                time_scope = datascope.get("time")
                # TODO
                if dataschema.is_timestamped:
                    pass
                elif dataschema.is_timespanned:
                    pass
                elif dataschema.is_timeblocked:
                    pass
            return True

        return validate
