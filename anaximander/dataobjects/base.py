from copy import copy
from typing import ClassVar
import pandas as pd

from ..descriptors import option, metamethod
from ..meta import Object, archetype


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

    def __str__(self):
        if hasattr(self, "_data"):
            return str(self._data)
        return super().__str__()
