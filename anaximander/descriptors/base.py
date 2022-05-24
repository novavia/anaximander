"""This module defines the base descriptor and registry classes."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
import dataclasses as dc
from itertools import chain
from types import new_class
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    ClassVar,
    TypeVar,
    Union,
)

from ..utilities.nxdataclasses import Validated

from .fieldtypes import UNDEFINED

Setting = tuple[str, Any]


class Descriptor(ABC):
    """The base class for Anaximander desriptors.

    Anaximander descriptors are used in type declarations to specify object
    attributes. These range from data and metadata fields to properties,
    relations, a named report etc.

    Descriptor collection is conducted by the metaclass, which by default
    merely indexes them in a purpose-built descriptor registry. The
    interpretation of descriptors collected in that registry is triggered
    by type decorators (notably @archetype and @datamodel). However
    descriptors are invoked in a third context, which is the initialization
    of new types by the metaclass. In that stage, the metaclass composes
    an archetype with a data model, whose registered descriptors are
    combined to determine the attributes and methods of the new type.

    Hence it is noteworthy that Anaximander descriptors do not
    in fact implement the Python descriptor protocol -because the attribute
    functionality is delegated to a different class of objects named Attribute.
    Attribute does implement the Python descriptor protocol as one might
    expect.

    The rationale for the delegation of the descriptor protocol is that
    Anaximander type declarations are composed -as in, say, a data archetype
    and a data model combined to form a certain type of data object. The
    attributes of the concrete data type depend on both the archetype and the
    model. Each declares descriptors, and these descriptors have to be
    combined by the metaclass to yield the data type.

    As a result, Descriptors serve as an interface for generating field types
    and schemas, and methods are written to that effect. Further, the
    architecture relies on multiple dispatchining (implemented with
    multimethod) to resolve the specifications stemming from structural
    archetypes, descriptor function and field type into the target
    implementation. This is done on module-level functions which can
    then be overriden with multimethod decorators for specific types that
    stand on a higher layer of the framework and cannot be imported here.

    The Descriptor interface provides class methods that can be used
    by a metaclass to collect descriptor declarations. Declaring Anaximander
    descriptors in a class without a metaclass that calls these class
    methods is technically possible but basically useless.
    """

    # A list of unallowed names for descriptors
    __reserved__: ClassVar[set[str]] = {
        # Reserved attribues
        "abstract",
        "anonymous",
        "archetype",
        "data",
        "metadata",
        "options",
        # Metaclass keyword arguments
        "type_name",
        "register_type",
        "overtype",
    }

    # ----- class variables ----- #
    # The global set of descriptor handles
    __handles__: ClassVar[set[str]] = set()
    # And a mapping of alternative handle names for convenience
    __handle_alternatives__: ClassVar[dict[str, str]] = dict()
    # The set of registry names that a given subclass registers into
    registries: ClassVar[set[str]] = set()
    # An optional handle for concrete classes
    handle: ClassVar[Optional[str]] = None
    # And the optional alternatives
    handle_alternatives: ClassVar[Optional[Iterable[str]]] = None
    # And the full list of relevant handles (inherited + alternatives)
    handles: ClassVar[set[str]] = set()
    # If this is True, then registries restore the output of the
    # _post_registry method to the namespace from which the descriptor is
    # collected
    mutate_after_registry: ClassVar[bool] = False

    # ----- instance attributes ----- #
    name: str

    def __init_subclass__(cls) -> None:
        mro = [parent for parent in cls.mro() if issubclass(parent, Descriptor)]
        handles = {r for p in mro if (r := getattr(p, "handle", None)) is not None}
        cls.registries = handles
        Descriptor.__handles__.update(handles)
        # Adds optional alternative handle designations
        if (handle := cls.handle) is not None and "handle_alternatives" in vars(cls):
            # This is only entered in cls declared its own alternatives
            alternatives = getattr(cls, "handle_alternatives")
            if alternatives is not None:
                alternatives = set(alternatives)
                for alternative in alternatives:
                    Descriptor.__handle_alternatives__[alternative] = handle
            else:
                alternatives = set()
        else:
            alternatives = set()
        if issubclass(cls, Validated):
            cls.__set_pydantic_model_class__()
        try:
            parent = mro[1]
        except IndexError:
            parent_handles = set()
        else:
            parent_handles = parent.handles
        cls.handles = parent_handles | handles | alternatives

    @classmethod
    def set_up_registry(cls, owner: type):
        """Sets up a descriptor registry on owner."""
        parent: Optional[DescriptorRegistry] = getattr(owner, "__descriptors__", None)
        registry = DescriptorRegistry(parent=parent)
        setattr(owner, "__descriptors__", registry)

    def __set_name__(self, owner: type, name: str):
        """Sets name by enforcing restrictions."""
        if name in self.__reserved__:
            msg = f"A metadescriptor's name cannot belong to {self.__reserved__}"
            raise AttributeError(msg)
        self.name = name

    @classmethod
    def __registry__(cls) -> dict[str, dict]:
        """A registry structure constructor."""
        return {handle: {} for handle in cls.__handles__}

    def _post_registry(self):
        """Returns a value that is set on the namespace from which a descriptor was collected."""
        return NotImplemented

    def set_attribute(self, host: type):
        """Sets an attribute from self on a receiving host type."""
        return NotImplemented


V = TypeVar("V")


class Registry(Mapping, Generic[V]):
    """Base class for DescriptorRegistry and SettingRegistry."""

    factory = Descriptor.__registry__

    parent: Optional["Registry"] = None
    _registry: dict[str, dict[str, V]] = dc.field(default_factory=factory)

    def __getitem__(self, key: str) -> dict[str, V]:
        try:
            return self._registry[key]
        except KeyError:
            try:
                handle = Descriptor.__handle_alternatives__[key]
                return self._registry[handle]
            except KeyError:
                msg = f"Unknown descriptor handle designation '{key}'"
                raise KeyError(msg)

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def __copy__(self):
        copy_ = type(self)(parent=self.parent)
        copy_._registry = {k: v.copy() for k, v in self._registry.items()}
        return copy_

    def handles(self, name, recursive=True) -> tuple[str]:
        """Returns the handles in which name is registered."""
        if recursive:
            registry = self.recursive._registry
        else:
            registry = self._registry
        return tuple(h for h, sub in registry.items() if name in sub)

    def pop_name(self, name: str):
        """Removes a descriptor name from all subregistries."""
        for sub in self._registry.values():
            sub.pop(name, None)

    def names(self, *handles, recursive=False) -> set[str]:
        """Returns a set of registered descriptor names."""
        if recursive:
            return self.recursive.names(*handles)
        else:
            handles = handles or Descriptor.__handles__
            subs = [self[handle] for handle in handles]
            return set(chain(*[sub.keys() for sub in subs]))

    def to_dict(self, recursive=False) -> dict[str, Descriptor]:
        """Returns a mapping name: descriptor from a registry

        In other words, this method removes the handle layer to return
        a single-depth mapping of names to descriptors.
        """
        if recursive:
            return self.recursive.to_dict()
        else:
            return dict(chain(*[sub.items() for sub in self._registry.values()]))

    @property
    def recursive(self) -> "Registry":
        """Returns an instance that is recursively merged with parents registries."""
        result = copy(self)
        if self.parent is None:
            return result
        parent = self.parent.recursive
        for name in self.names():
            parent.pop_name(name)
        merged = {h: parent[h] | self[h] for h in Descriptor.__handles__}
        result._registry = merged
        return result

    def _register(self, name, value, *handles):
        """Primitive for register."""
        # We remove the name from the registry
        self.pop_name(name)
        for handle in handles:
            self._registry[handle][name] = value

    @abstractmethod
    def register(self, *args, **kwargs):
        """Registers a new item."""
        return NotImplemented

    @abstractmethod
    def collect(self, namespace: Mapping, *handles, strip: bool = False):
        """Collects and registers items from a namespace.

        Handles limit collection to the specified handles.
        If strip is True, the method attempts to delete the collected items
        from the namespace -which may raise a TypeError if the object does
        not support item deletion.
        """
        return NotImplemented

    def fetch(
        self,
        *handles: str,
        recursive: bool = False,
        filter: Optional[Callable[[V], bool]] = None,
        **kwargs,
    ) -> dict[str, V]:
        """Returns a flat mapping of registered values

        Arguments:
            handles: optional registration strings (e.g. "metadata", "options")
            This will restrict the search to those subregistries only. Otherwise the
            entire registry is searched.
            recursive: if True, parent registries are searched recursively
            filter: an optional lambda function that is applied to values.
            Only the values that evaluate True are returned.
            **kwargs: if supplied, works as a shortcut to filter by applying
            an attribute lookup on registered items. The key is the attribute
            name and the value can be either the target value of the attribute,
            or a tuple or list thereof. If the target value itself must be
            a tuple or list, use filter instead.

            Any number of kwargs are joined along with filter by applying an
            "AND" condition.
        """
        registry = self.recursive if recursive else self
        results = {}
        filters = []
        if filter is not None:
            filters.append(filter)
        for k, v in kwargs.items():
            if isinstance(v, (tuple, list)):
                filters.append(lambda any: getattr(any, k, UNDEFINED) in v)
            else:
                filters.append(lambda any: getattr(any, k, UNDEFINED) == v)
        if not handles:
            handles = Descriptor.__handles__
        for handle in handles:
            for name, value in registry[handle].items():
                if name in results:
                    continue
                elif all(f(value) for f in filters):
                    results[name] = value
        return results

    def __str__(self):
        dataprint = dict()
        for k, sub in self._registry.items():
            dataprint[k] = {k: str(v) for k, v in sub.items()}
        return str(dataprint)
        # return yaml.dumps(dataprint)


@dataclass(repr=False)
class DescriptorRegistry(Registry[Descriptor]):
    """A descriptor registries, organized into class-defined handles.

    Descriptor collection is conducted into a dedicated DescriptorRegistry
    instance. These registries are organized as a two-layer string-keyed
    dictionary, whose first entry key points to a type of descriptors,
    and the the second entry is the descriptor name. Name unicity is
    nevertheless guaranteed across descriptor types, that is, only one descriptor
    of a given name can reside in the registry. The first-level dictionary keys
    are a lower-case handle that concrete Descriptor classes must declare and
    also serves as the programming interface (e.g. 'data', 'option', etc.).

    Where there are inheritance relationships between descriptor classes,
    descriptors get registered in all subregistries -hence a MetaField gets
    registered in both the 'metadata' and 'metafield' subregistries. However the
    Descriptor registry provides a fetch interface that removes duplicates in
    interations that span across types.

    At instantiation a descriptor registry can declare a parent registry. As
    a result, the fetch inteface offers to flavors: non-recursive or recursive.
    In the non-recursive version, only the descriptors directly registered
    with the present registry instance are returned. In the recursive version,
    the search also inludes the parent registry, and so recursively. The results
    are merged through a dictionary merge operation, with the higher caller
    in the stack taking precedence.
    """

    factory = Descriptor.__registry__

    parent: Optional["DescriptorRegistry"] = None
    _registry: dict[str, dict[str, Descriptor]] = dc.field(default_factory=factory)

    def collect(
        self,
        namespace: Union[type, Mapping, "DescriptorRegistry"],
        *handles: str,
        strip: bool = False,
    ):
        """Collects and registers descriptors from a namespace, alernatively an other registry.

        :param namespace: a mapping or a class from which to collect descriptors
        :param handle: limits collection to the specified handles
        :param strip: if True, the method attempts to delete the collected items
            from the namespace -which may raise a TypeError if the object does not
            support item deletion.
        """
        if isinstance(namespace, type):
            namespace = namespace.__dict__
        elif isinstance(namespace, DescriptorRegistry):
            namespace = namespace.to_dict()
        handles = handles or Descriptor.__handles__
        # Creates a mock class that calls __set_name__ on descriptors
        exec_body = lambda ns: ns.update(namespace)
        new_class("Owner", exec_body=exec_body)
        for name, attr in dict(namespace).items():
            if isinstance(attr, Descriptor):
                if any(h in handles for h in attr.handles):
                    self.register(attr, name)
                    if strip:
                        del namespace[name]
                    if attr.mutate_after_registry:
                        try:
                            namespace[name] = attr._post_registry()
                        except TypeError:
                            pass

    def register(self, descriptor: Descriptor, name: str):
        """Registers a descriptor."""
        dtype = type(descriptor)
        if not issubclass(dtype, Descriptor):
            msg = f"{self} can only register descriptors, not {dtype.__name__}"
            raise TypeError(msg)
        handles = dtype.registries
        self._register(name, descriptor, *handles)

    def __repr__(self):
        return "<DescriptorRegistry>"


@dataclass(repr=False)
class SettingRegistry(Registry[Any]):
    """A registry that buffers values to be set onto descriptors.

    A SettingRegistry instance points to a DescriptorRegistry, whose structure
    it accesses in order to register settings.
    """

    factory = Descriptor.__registry__

    descriptors: DescriptorRegistry
    parent: Optional["SettingRegistry"] = None
    _registry: dict[str, dict[str, Any]] = dc.field(default_factory=factory)

    def collect(
        self,
        namespace: Union[type, Mapping, "SettingRegistry"],
        *handles,
        strip: bool = False,
    ):
        """Collects and registers settings from a namespace, alternatively another SettingRegistry.

        Handles limits collection to those specified handles.
        If strip is True, the method attempts to delete the collected items
        from the namespace -which may raise a TypeError if the object does
        not support item deletion.
        """
        targets = self.descriptors.names(*handles, recursive=True)
        if isinstance(namespace, type):
            namespace = type.__dict__
        elif isinstance(namespace, SettingRegistry):
            namespace = namespace.to_dict()
        for k, v in dict(**namespace).items():
            if k in targets:
                self.register(k, v)
                if strip:
                    del namespace[k]

    def register(self, *setting: Setting):
        """Registers a key, value pair aka. a Setting."""
        name, value = setting
        handles = self.descriptors.handles(name)
        self._register(name, value, *handles)

    def __copy__(self):
        copy_ = type(self)(self.descriptors, parent=self.parent)
        copy_._registry = {k: v.copy() for k, v in self._registry.items()}
        return copy_

    def __repr__(self):
        return "<SettingRegistry>"


class DataDescriptor(Descriptor):
    """Mix-in class for data descriptors, which make up data models."""

    handle = "data"


class MetaDescriptor(Descriptor):
    """Mix-in class for meta descriptors, which make up object archetypes."""

    pass


class DataModelSpecifier:
    """Base class for data types specifiers Field, Schema, IndexSchema, and TypeSpec."""

    @classmethod
    def __get_validators__(cls_):
        # Enables type checking by Pydantic
        def check_type(cls, value):
            if not isinstance(value, cls_):
                raise TypeError(f"A {cls_.__name__} instance is required")
            return value

        yield check_type
