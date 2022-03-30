"""This module implements attributes set on framework types and objects."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar, Generic, Optional

from ..utilities.functions import attribute_owner
from ..utilities.nxdataclasses import Validated

from .dataspec import DataSpec

T = TypeVar("T")
Getter = Callable[[Any], T]


class AttributeBase(Generic[T]):
    pass


@dataclass
class Attribute(Validated, AttributeBase):
    """A specialized descriptor that mimics an object attribute.

    attributes:
        name: the name of the attribute
        retriever: the method by which the attribute is retrieved in a host
            object
        transformer: an optional transformation method for retrieved data
        aggregator: an optional aggregation method for data retrieved on a host
            object that implements a collection interface
        getter: the attribute getter, computed after initialization based on the
            retriever, transformer and aggregator specs
        _field: an optional attribute that points back to a Field object from
            which the attribute was generated. This is used as a convenience
            for metaprogramming functions.
    """

    name: str
    retriever: Callable[[Any], Any] = field(repr=False)
    transformer: Optional[Getter[T]] = field(default=None, repr=False)
    aggregator: Optional[Callable[[Iterable], T]] = field(default=None, repr=False)
    getter: Getter[T] = field(init=False, repr=False)
    _field: Any = None

    def __post_init__(self):
        if self.transformer is None and self.aggregator is None:
            self.getter = self.retriever
        elif self.aggregator is None:
            self.getter = lambda obj: self.transformer(self.retriever(obj))
        elif self.transformer is None:
            self.getter = lambda obj: self.aggregator(self.retriever(obj))
        else:
            self.getter = lambda obj: self.aggregator(
                self.transformer(self.retriever(obj))
            )
        Validated.__post_init__(self)

    def __get__(self, obj, cls=None) -> T:
        if obj is None:
            return self
        return self.getter(obj)

    def __set__(self, obj, value):
        # Attribute handles the special case in which it is set on a metaclass
        # as well as on the classes generated from that metaclass but one
        # of these classes needs to override the attribute (which is basically
        # an instance property).
        if isinstance(obj, type):
            if isinstance(value, (Attribute, property)):
                # The attribute is being overloaded
                metaclass_owner = attribute_owner(type(obj), self.name)
                # In order to reset the attribute, we have to
                # temporarily eliminate it from the metaclass that declares
                # it -otherwise setattr enters an infinite recursion.
                delattr(metaclass_owner, self.name)
                setattr(obj, self.name, value)
                setattr(metaclass_owner, self.name, self)
                return
        msg = f"Cannot set read-only attribute {self.name}"
        raise AttributeError(msg)

    def __delete__(self, obj):
        msg = f"Cannot delete read-only attribute {self.name}"
        raise AttributeError(msg)

    def __typespec__(self, arc: type) -> DataSpec:
        dataspec = getattr(self._field, "dataspec")()
        # This is hacky but circumvents circular import issues
        # A cleaner alternative would be to monkeypatch the method
        key = getattr(getattr(arc, "metatype"), "__unset_typespec_metadata__")[0]
        return (), {key: dataspec}
