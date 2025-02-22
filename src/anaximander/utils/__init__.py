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
    """Facility to hold keyword arguments and pass them to a function or stringify them.

    The __init__ method accepts a context argument, typically an object from which attributes
    will be looked up. If the context is None, then locals() will serve as the default context.
    The interface accepts the special syntax ".name" in the values to indicate that
    the kwarg map will look for an attribute named "name" in the context. A further shortcut
    is to pass a list of argument names to the special key "_", such that:
        KwargMap(obj, _=["a", "b"])
    is equivalent to:
        KwargMap(obj, a=".a", b=".b")
    and exposes the mapping {"a": obj.a, "b": obj.b}
    KwargMap also implements key sorting, preserving insertion order by default. The sort
    method will modify sorting in place.
    Finally, KwargMap implements __call__, which by default will return a new instance of
    KwargMap with the same attributes. However the call can be used to specify a subset of
    keys, in order, hence operating a subselection.
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
        """Primitive for looking up keys in the context."""
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
        """Sorts the KwargMap according to a specified key sequence.

        Args:
            key: A sequence of keys defining the desired order. Keys in the mapping not present
                in the sequence will be appended at the end in their original order. Keys in the
                sequence but not in the mapping will raise an error. Alternatively, takes the
                same argument as list.sort(), i.e. a function of one argument used to extract a
                comparison key from each list element.
            reverse: Whether to sort in descending order.
            _remove_missing_keys: used by the __call__ method to eliminate keys not supplied in
                case key is a sequence of strings.
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
        """Returns a new instance, restricted to the specified keys, and sorted accordingly."""
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
    """A structured configuration class."""

    @property
    def omegaconf(self) -> DictConfig:
        return OmegaConf.structured(self)

    @classmethod
    def load[C](cls: type[C], path: Path | str) -> C:
        path = Path(path)
        return cls(**OmegaConf.load(path))  # type: ignore

    def save(self, path: Path | str):
        path = Path(path)
        with open(path, "w") as f:
            OmegaConf.save(config=self, f=f)

    def __str__(self):
        return OmegaConf.to_yaml(self)
