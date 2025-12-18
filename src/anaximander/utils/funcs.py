"""General-purpose utilities for environment detection, string manipulation, and iteration helpers."""
import contextlib
import itertools
import os
import re
import socket
from pathlib import Path
from types import ModuleType
from typing import Generator


import inflect
from inflect import Word

IE = inflect.engine()


def boolean(string):
    """Convert a string literal to a boolean.

    Args:
        string (str): The string "True" or "False".

    Returns:
        bool: The corresponding boolean value.

    Raises:
        ValueError: If the input is not "True" or "False".
    """
    if string == "True":
        return True
    elif string == "False":
        return False
    else:
        raise ValueError


def is_online():
    """Check if outbound network connectivity is available.

    Returns:
        bool: True if a connection to a well-known host can be made, False otherwise.
    """
    connection = None
    try:
        host = socket.gethostbyname("www.google.com")
        connection = socket.create_connection((host, 80), 2)
    except:  # noqa
        return False
    else:
        return True
    finally:
        if connection is not None:
            connection.close()


def offline(assertion=None):
    """Get or set the offline status.

    If assertion is provided, the offline status is set accordingly. When setting
    to False, an online connectivity check is performed before updating the status.

    Args:
        assertion (bool | None): Desired offline status. If None, the current status is returned.

    Returns:
        bool: Current offline status.
    """
    if assertion is not None:
        if assertion is False:
            try:
                assert is_online()
            except AssertionError:
                pass
            else:
                os.environ["OFFLINE"] = "False"
                return False
        os.environ["OFFLINE"] = "True"
        return True
    try:
        return boolean(os.environ["OFFLINE"])
    except KeyError:
        which = not is_online()
        os.environ["OFFLINE"] = str(which)
        return which


def local_runtime() -> bool:
    """Indicate whether the application is running locally.

    Returns:
        bool: True if a local runtime is detected, False otherwise.
    """
    return bool(os.getenv("IS_RUNNING_LOCALLY"))


def is_close(a, b, tolerance=1e-9):
    """Test near-equality of two numbers within a tolerance.

    Args:
        a (float): First value.
        b (float): Second value.
        tolerance (float, optional): Maximum allowed absolute difference. Defaults to 1e-9.

    Returns:
        bool: True if the absolute difference is less than tolerance.
    """
    return abs(a - b) < tolerance


def batched(iterable, n: int):
    """Yield fixed-size batches from an iterable.

    Args:
        iterable (Iterable): Source of items.
        n (int): Batch size. Must be at least 1.

    Yields:
        tuple: Tuples of up to n items from the iterable.

    Raises:
        ValueError: If n is less than 1.
    """
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


def subclasses(cls, depth: int = -1, strict: bool = True) -> list[type]:
    """Recursively get all subclasses of a class up to a given inheritance depth.

    Args:
        cls (type): The base class.
        depth (int, optional): Maximum depth to traverse. Defaults to -1 for no limit.
        strict (bool, optional): If True, exclude cls itself from the results. Defaults to True.

    Returns:
        list[type]: All discovered subclasses.
    """
    all = []
    if not strict:
        all.append(cls)
    if depth != 0:
        subs = cls.__subclasses__()
        all.extend(subs)
        for sub in subs:
            all.extend(subclasses(sub, depth=depth - 1))
    return all


def camel_to_snake(s: str) -> str:
    """Converts a camel case string to a snake case string.

    Args:
        s (str): string to convert.

    Returns:
        str: converted string.
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    s2 = re.sub("__([A-Z])", r"_\1", s1)
    s3 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s2)
    return s3.lower()


def pluralize(s: str) -> str:
    """Pluralizes a string.

    Args:
        s (str): string to pluralize.

    Returns:
        str: pluralized string.
    """
    if isinstance(s, Word):
        if not IE.singular_noun(s):
            return IE.plural(s)
    return s


def type_name_to_collection_name(name: str) -> str:
    """Converts a class name to a collection name.

    Effectively converts CamelCase to snake_case and adds plural form.

    Args:
        name (str): A class name presumed to be CamelCase.

    Returns:
        str: A collection name using snake_case and plural form.
    """
    snake = camel_to_snake(name)
    *prefixes, last_word = snake.split("_")
    if isinstance(last_word, Word):
        last_word = pluralize(last_word)
    return "_".join(prefixes + [last_word])


@contextlib.contextmanager
def workdir(path: Path | str, *, mkdir: bool = True) -> Generator[Path, None, None]:
    """Temporarily change the working directory.

    Optionally creates the directory before entering.

    Args:
        path (Path | str): Target directory.
        mkdir (bool, optional): Whether to create the directory if it does not exist. Defaults to True.

    Yields:
        Path: The path of the active working directory.

    Raises:
        ValueError: If the path exists but is not a directory.
    """
    path = Path(path)
    cwd = Path.cwd()
    if mkdir:
        if not path.exists():
            path.mkdir(parents=True)
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory.")
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def is_package_init(module: ModuleType):
    """Return whether the module corresponds to a package __init__ file.

    Args:
        module (ModuleType): Module to inspect.

    Returns:
        bool: True if the module file name is '__init__.py', False otherwise.
    """
    if not (file_path := getattr(module, '__file__', None)):
        return False
    path = Path(file_path)
    return path.stem == "__init__"
