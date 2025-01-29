import re

import inflect
from inflect import Word

IE = inflect.engine()


def subclasses(cls, depth: int = -1, strict: bool = True) -> list[type]:
    """Recursively get all subclasses of a class, to a given inheritance depth.

    Args:
        depth (int, optional): Sets the inheritance depth. Defaults to -1,
            which means all subclasses.
        strict (bool, optional): If True, cls itself is excluded from the results.

    Returns:
        list[type]: An list of subclasses.
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
