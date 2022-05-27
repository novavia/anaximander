from collections.abc import Mapping
import dataclasses
from typing import Optional, Union

from pydantic import BaseModel
import pandas as pd

from anaximander.descriptors.fields import DataField

from ..utilities.nxdataclasses import DataClass
from ..descriptors.base import DescriptorRegistry
from ..descriptors.schema import DataSchema
from ..descriptors import metadata, metamethod, metaproperty

from ..meta import archetype, metamorph, trait

from .base import DataObject
from .data import Data


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
            archetype = getattr(field.type_, "archetype", Data)
            metadata = dataspec.extensions.get("metadata", {})
            fieldtype = archetype.subtype(dataspec=dataspec, **metadata)

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
    def has_entity_schema(schema: DataSchema):
        """Schema must be an entity schema"""
        return schema.is_entity_schema


@metamorph
@archetype
class Record(Model, metacharacters=["record"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def has_record_schema(schema: DataSchema):
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
    def has_atemporal_schema(schema: DataSchema):
        """Schema must be atemporal"""
        return not schema.is_temporal


@archetype
class Sample(Record, metacharacters=["timestamped"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def has_timestamped_schema(schema: DataSchema):
        """Schema must feature a timestamp"""
        return schema.is_timestamped


@archetype
class Event(Record, metacharacters=["timestamped"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def has_timestamped_schema(schema: DataSchema):
        """Schema must feature a timestamp"""
        return schema.is_timestamped


@archetype
class Transition(Event):
    pass


@archetype
class Session(Record, metacharacters=["timespanned"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def has_timespanned_schema(schema: DataSchema):
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
    def has_timestamped_schema(schema: DataSchema):
        """Schema must feature a period"""
        return schema.is_timeblocked


@metamorph
@archetype
class Spec(Model, metacharacters=["spec"]):
    dataschema: DataSchema = metadata(typespec=True)

    @dataschema.validator
    def has_spec_schema(schema: DataSchema):
        """Schema cannot feature an index"""
        return schema.is_spec_schema
