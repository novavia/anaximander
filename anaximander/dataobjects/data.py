from typing import ClassVar, Optional, Union
from attr import has

from frozendict import frozendict

from ..descriptors.base import DescriptorRegistry, SettingRegistry
from ..descriptors.datatypes import nxdata, nxmodel
from ..descriptors.dataspec import DataSpec
from ..descriptors.schema import IndexSchema
from ..descriptors.scope import DataScope
from ..descriptors import (
    metadata,
    metamethod,
    _dataspec_keys,
    _pydantic_validator_keys,
)

from ..meta import archetype

from .base import DataObject


NoneType = type(None)

DataType = Union[type[nxdata], type[nxmodel], type[DataObject]]  # type: ignore


@archetype
class Data(DataObject):
    dataspec: DataSpec = metadata(typespec=True)
    dataindex: Optional[dict] = metadata(objectspec=True)

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
        base_settings = getattr(cls, "__settings__")
        descriptor_registry = DescriptorRegistry(parent=base_descriptors)
        descriptor_registry.collect(namespace, "data", strip=True)
        settings_registry = SettingRegistry(descriptor_registry, parent=base_settings)
        settings_registry.collect(namespace, "metadata")
        extensions = {k: v for k, v in namespace.items() if k in _dataspec_keys}
        extensions["metadata"] = settings_registry.fetch("metadata")
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

    # TODO: note that it is redundant to conform data that has already been
    # parsed. Optimization would require to rewrite __init__ or add a flag
    # to the Data class so that the logic can be handled in DataObject.__init__
    @metamethod
    def conformer(cls, dataspec: DataSpec, **metadata):
        model_cls = dataspec.model_class(namespace=cls, parse=False, validate=False)

        @classmethod
        def conform(cls_, data, **metadata):
            return model_cls(data=data).data

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

    def __getattr__(self, name):
        if isinstance(self.dataindex, dict):
            try:
                return self.dataindex[name]
            except KeyError:
                pass
        raise AttributeError(f"{type(self)} object has no attribute '{name}'")

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

    def __str__(self):
        if hasattr(self, "_data"):
            return f"{self._data} {self.unit}"
        return super().__str__()
