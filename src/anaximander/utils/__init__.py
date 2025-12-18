"""Utilities for keyword argument mapping and structured configuration support.

Provides KwargMap for contextual kwarg resolution and ordering, and Config
for attrs-based structured configurations compatible with OmegaConf.
"""

import functools
import inspect
from collections import UserDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, dataclass_transform

import attrs
from omegaconf import OmegaConf, DictConfig

from .meta import AutoDecoratedType


class KwargMap(UserDict[str, Any]):
    """Container for keyword arguments with contextual lookup and ordering.

    Supports contextual resolution using a leading dot in values (e.g., ".name"),
    which resolves against the provided context object or mapping. The special key
    "_" accepts a list of argument names as a shorthand, e.g.:
      KwargMap(obj, _=["a", "b"])
    is equivalent to:
      KwargMap(obj, a=".a", b=".b")
    producing {"a": obj.a, "b": obj.b}.

    Maintains insertion order and provides in-place sorting via sort(). Calling
    the instance returns a new KwargMap restricted to a subset of keys, ordered
    as provided.

    Args:
        context (Any, optional): Source for resolving ".name" references, typically
            an object or mapping. If None, the caller's locals() are used.
        *args: Additional positional arguments forwarded to UserDict.
        **kwargs: Initial mapping content.
    """

    def __init__(self, context: Any = None, *args: Any, **kwargs: Any):
        if context is None:
            caller_frame = inspect.stack()[1]
            context = caller_frame.frame.f_locals
        self.context = context
        self._key_sequence: list[str] = []  # type: ignore
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        if not isinstance(key, str):
            raise TypeError("Keys must be strings")
        _keys: list[str] = list(self.data.get("_", []))
        if key == "_":
            try:
                item = tuple(item)
                assert all(isinstance(i, str) for i in item)
                if not item:
                    return
            except (ValueError, TypeError, AssertionError):
                raise TypeError("_ only maps to a sequence of strings.")
            for k in _keys:
                self._key_sequence.remove(k)
        elif key in _keys:
            _keys.remove(key)
            self._key_sequence.remove(key)
            if _keys:
                self.data["_"] = tuple(_keys)
            else:
                del self.data["_"]
        super().__setitem__(key, item)
        if key not in self._key_sequence:
            if key == "_":
                self._key_sequence.extend(item)
            else:
                self._key_sequence.append(key)

    def _getitem_from_context(self, actual_key: str):
        """Look up a key in the context.

        Args:
            actual_key (str): Attribute or mapping key to resolve.

        Returns:
            Any: The resolved value from the context.

        Raises:
            AttributeError: If context is not a mapping and the attribute is missing.
            KeyError: If context is a mapping and the key is missing.
        """
        try:
            return getattr(self.context, actual_key)
        except AttributeError:
            if isinstance(self.context, Mapping):
                return self.context[actual_key]
            else:
                raise

    def __getitem__(self, key: str) -> Any:
        try:
            value = self.data[key]
            if isinstance(value, str) and value.startswith("."):
                real_key = value[1:]
                return self._getitem_from_context(real_key)
            else:
                return value
        except KeyError:
            if key in self.data.get("_", []):
                return self._getitem_from_context(key)
            else:
                raise

    def __delitem__(self, name: str) -> None:
        if name == "_":
            keys = self.data.pop("_")
            for key in keys:
                self._key_sequence.remove(key)
            return
        if name not in self._key_sequence:
            raise KeyError(name)
        _keys = list(self.data.get("_", []))
        if name in _keys:
            _keys.remove(name)
            if _keys:
                self.data["_"] = tuple(_keys)
            else:
                del self.data["_"]
        else:
            del self.data[name]
        self._key_sequence.remove(name)

    def __iter__(self):
        return iter(self._key_sequence)

    def __len__(self):
        return len(self._key_sequence)

    def sort(
        self,
        key: Sequence[str] | Callable[[str], Any] | None = None,
        reverse: bool = False,
        _remove_missing_keys: bool = False,
    ) -> None:
        """Sort keys according to a sequence or a key function.

        Args:
            key (Sequence[str] | Callable[[str], Any] | None): Either a sequence of
                keys defining the desired order or a function as in list.sort().
                Keys not in the sequence are appended in original order. Unknown
                keys in the sequence raise a KeyError.
            reverse (bool): Whether to sort in descending order.
            _remove_missing_keys (bool): Internal flag used by __call__ to drop keys
                not supplied when key is a sequence.

        Raises:
            KeyError: If key is a sequence containing unknown keys.
        """
        if isinstance(key, Sequence):
            key_sequence = list(key)
            if extra_keys := set(key_sequence) - set(self._key_sequence):
                msg = f"Cannot sort with unknown keys: {extra_keys}"
                raise KeyError(msg)
            keys_not_in_sequence = [k for k in self._key_sequence if k not in key_sequence]
            if _remove_missing_keys:
                for k in keys_not_in_sequence:
                    del self[k]
                self._key_sequence = key_sequence
            else:
                new_key_sequence = key_sequence + keys_not_in_sequence
                self._key_sequence = new_key_sequence
            if reverse:
                self._key_sequence.reverse()
        else:
            self._key_sequence.sort(key=key, reverse=reverse)

    def __call__(self, *keys):
        """Return a new instance restricted to the specified keys.

        Args:
            *keys: Optional ordered keys to include in the new instance.

        Returns:
            KwargMap: A copy including only the specified keys, ordered accordingly.
        """
        copy = type(self)(self.context, **self.data)
        if keys:
            copy.sort(keys, _remove_missing_keys=True)
        return copy

    def __repr__(self):
        return f"{self.__class__.__name__}(context={self.context}, data={self.data})"

    def __str__(self):
        return ", ".join(f"{k}={v}" for k, v in self.items())


_config_decorator = functools.partial(attrs.define, auto_attribs=True)


class ConfigType(AutoDecoratedType, decorator=_config_decorator):
    pass


@dataclass_transform()
class Config(metaclass=ConfigType):
    """Structured configuration base compatible with OmegaConf.

    Classes deriving from Config are attrs-structured via the AutoDecoratedType
    metaclass and can be saved/loaded using OmegaConf.
    """

    @property
    def omegaconf(self) -> DictConfig:
        """Return an OmegaConf structured view of this instance.

        Returns:
            DictConfig: Structured configuration built from this instance.
        """
        return OmegaConf.structured(self)

    @classmethod
    def load[C](cls: type[C], path: Path | str) -> C:
        """Load a configuration from a file.

        Args:
            path (Path | str): Path to a file readable by OmegaConf.

        Returns:
            C: An instance of the configuration class populated from the file.
        """
        path = Path(path)
        return cls(**OmegaConf.load(path))  # type: ignore

    def save(self, path: Path | str):
        """Save the configuration to a file.

        Args:
            path (Path | str): Target path to write the configuration.
        """
        path = Path(path)
        with open(path, "w") as f:
            OmegaConf.save(config=self, f=f)

    def __str__(self):
        """Return a YAML representation of the configuration."""
        return OmegaConf.to_yaml(self)


private_field = functools.partial(attrs.field, init=False, repr=False, eq=False, order=False)
