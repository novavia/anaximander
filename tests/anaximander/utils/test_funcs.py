import sys

import anaximander as nx
from anaximander.utils.funcs import (
    camel_to_snake,
    is_package_init,
    pluralize,
    subclasses,
    type_name_to_collection_name,
)


class C0:
    pass


class C1(C0):
    pass


class D1(C0):
    pass


class D2(D1):
    pass


def test_subclasses():
    assert subclasses(C0) == [C1, D1, D2]
    assert subclasses(C0, strict=False) == [C0, C1, D1, D2]
    assert subclasses(C0, depth=0) == []
    assert subclasses(C0, depth=1) == [C1, D1]
    assert subclasses(C0, depth=2) == [C1, D1, D2]
    assert subclasses(C0, depth=3) == [C1, D1, D2]


def test_camel_to_snake():
    assert camel_to_snake("camelCase") == "camel_case"
    assert camel_to_snake("CamelCase") == "camel_case"
    assert camel_to_snake("camel_case") == "camel_case"
    assert camel_to_snake("getHTTPResponseCode") == "get_http_response_code"
    assert camel_to_snake("") == ""


def test_pluralize():
    assert pluralize("word") == "words"
    assert pluralize("Words") == "Words"
    assert pluralize("index") == "indexes"


def test_type_name_to_collection_name():
    assert type_name_to_collection_name("CamelCase") == "camel_cases"
    assert type_name_to_collection_name("HTTPResponseCode") == "http_response_codes"
    assert type_name_to_collection_name("CamelCaseThing") == "camel_case_things"


def test_is_package_init():
    this_module = sys.modules[__name__]
    assert not is_package_init(this_module)
    assert is_package_init(nx)
