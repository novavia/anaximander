"""This module defines the TypeSpec class.

Type specifications, named TypeSpec in the program, serve to specify and
uniquely identify synthetic types that are generated throught the Anaximander
machinery.
A type specification includes a reference to an archetype, a set of
metacharacters and a dictionary mapping type parameters to values. The
TypeSpecification object has built-in serialization / deserialization in
JSON (pending) and YAML, and it prints to a YAML-formatted output.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Union

from frozendict import frozendict
from ruamel.yaml.comments import CommentedMap

from ..utilities.nxdataclasses import Validated
from ..utilities.nxyaml import yaml_object, yaml


@yaml_object(yaml)
@dataclass(frozen=True, repr=False)
class TypeSpec(Validated):
    """Input to Anaximander's type factory.

    A TypeSpec contains an archetype, a set of metacharacter strings and a dictionary
    of type mapping metadata parameters to values. These are the arguments necessary
    to generate a type in the Anaximander's archetype machinery.
    TypeSpec supports two operations related to archetypes:
    - normalize: archetypes are linked by inheritance and supplying a
    metacharacter to an archetype may lead to another archetype -in case the derived
    archetype is registered as a metamorphism of the parent archetype. The
    normalize method reduces a specification by traversing metamorphic inheritance
    chains as necessary, returning a spec that indicates the resolved implementation
    archetype, along with metacharacters and metadata -which may be
    stripped through the normalization process.
    TODO: provide an example
    - validate: validates that a type specification is valid by confirming
    that the metacharacters and metadata can be interpreted by the
    archetype. Normalize is called first, and validation code is run on the normal
    form.
    """

    yaml_tag: ClassVar[str] = "!typespec"

    archetype: type
    metacharacters: frozenset[str]
    metadata: frozendict[str, Any]

    def __init__(
        self,
        archetype: type,
        *metacharacters: str,
        **metadata: Any,
    ):
        object.__setattr__(self, "archetype", archetype)
        metacharacters = frozenset(metacharacters)
        object.__setattr__(self, "metacharacters", metacharacters)
        object.__setattr__(self, "metadata", frozendict(metadata))
        Validated.__post_init__(self)

    def normalize(self) -> "TypeSpec":
        """Normalizes self by resolving archetype metamorphic chains if necessary."""
        spec_interpreter = getattr(self.archetype, "__typespec_interperter__")
        return spec_interpreter(*self.metacharacters, **self.metadata)

    def validate(self) -> bool:
        """Checks attributes integrity through cross-referencing.

        If archetype is supplied, the method confirms its equality to
        self's archetype attribute.
        """
        # TODO: should the method also validate the values given to type parameters?
        # Maybe add a flag T/F for it?
        try:
            self = self.normalize()
        except TypeError:
            return False
        archetype: type = self.archetype
        metacharacters = getattr(archetype, "metacharacters")
        metadata = getattr(archetype, "metadata")
        try:
            assert all(m in metacharacters for m in self.metacharacters)
            assert all(n in metadata.fields for n in self.metadata)
        except AssertionError:
            return False
        else:
            return True

    def __le__(self, other: "TypeSpec"):
        archetype_test = issubclass(other.archetype, self.archetype)
        metacharacters_test = self.metacharacters <= other.metacharacters
        metadata_test = self.metadata.items() <= other.metadata.items()
        return archetype_test and metacharacters_test and metadata_test

    def __lt__(self, other: "TypeSpec"):
        return self <= other and self != other

    def __gt__(self, other: "TypeSpec"):
        return self >= other and self != other

    def __ge__(self, other: "TypeSpec"):
        archetype_test = issubclass(self.archetype, other.archetype)
        metacharacters_test = self.metacharacters >= other.metacharacters
        metadata_test = self.metadata.items() >= other.metadata.items()
        return archetype_test and metacharacters_test and metadata_test

    def __and__(self, other: "TypeSpec") -> "TypeSpec":
        if issubclass(other.archetype, self.archetype):
            archetype = self.archetype
        elif issubclass(self.archetype, other.archetype):
            archetype = other.archetype
        else:
            return NotImplemented
        metacharacters = self.metacharacters & other.metacharacters
        metadata = dict(self.metadata.items() & other.metadata.items())
        return typespec(archetype, *metacharacters, **metadata)

    def __or__(self, other: "TypeSpec") -> "TypeSpec":
        if issubclass(other.archetype, self.archetype):
            archetype = other.archetype
        elif issubclass(self.archetype, other.archetype):
            archetype = self.archetype
        else:
            return NotImplemented
        metacharacters = self.metacharacters | other.metacharacters
        metadata = dict(self.metadata.items() | other.metadata.items())
        return typespec(archetype, *metacharacters, **metadata)

    def __sub__(self, other: "TypeSpec") -> "TypeSpec":
        if issubclass(self.archetype, other.archetype):
            archetype = self.archetype
            metacharacters = self.metacharacters - other.metacharacters
            metadata = dict(self.metadata.items() - other.metadata.items())
            return typespec(archetype, *metacharacters, **metadata)
        return NotImplemented

    @classmethod
    def to_yaml(cls, representer, node: "TypeSpec"):
        archetype = node.archetype.__name__
        metacharacters = list(node.metacharacters)
        metadata = dict(node.metadata)
        data = dict(
            archetype=archetype,
            metacharacters=metacharacters,
            metadata=metadata,
        )
        return representer.represent_mapping(cls.yaml_tag, data)

    @classmethod
    def from_yaml(cls, constructor, node):
        # TODO: the archetype needs to be retrieved from its string representation
        data = CommentedMap()
        constructor.construct_mapping(node, data, deep=True)
        archetype = data.pop("archetype", None)
        metacharacters = data.pop("metacharacters", [])
        metadata = data.pop("metadata", {})
        return cls(archetype, *metacharacters, **metadata)

    def __repr__(self):
        string = f"<typespec:{self.archetype.__name__}"
        if self.metacharacters:
            string += "|" + ", ".join(self.metacharacters)
        if self.metadata:
            string += "|" + ", ".join(
                [f"{k}={repr(v)}" for k, v in self.metadata.items()]
            )
        string += ">"
        return string

    def __str__(self):
        return yaml.dumps(self)


def typespec(
    archetype: Union[type, TypeSpec],
    *metacharacters: str,
    **metadata: Any,
) -> TypeSpec:
    """Helper function to instantiate TypeSpec objects.

    If the first argument is a TypeSpec instance, then the function simply
    passes it on.
    """
    if isinstance(archetype, TypeSpec):
        return archetype
    return TypeSpec(archetype, *metacharacters, **metadata)
