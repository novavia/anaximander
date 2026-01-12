from typing import Any

import pytest

from anaximander.aml.declarative import AnnotatableDeclarator, Declarator, EnumerationDeclarator


def test_annotatable_infers_types_from_generic_argument():
    class NumericAnnotator(AnnotatableDeclarator[int]):
        def __validate_type__(self, type: Any) -> bool:  # noqa: A003
            return super().__validate_type__(type)

    assert NumericAnnotator.__types__ == (int,)


def test_enumeration_infers_member_types_from_generic_argument():
    class FieldDeclarator(AnnotatableDeclarator):
        __types__ = (Declarator,)

        def __validate_type__(self, type: Any) -> bool:  # noqa: A003
            return super().__validate_type__(type)

    class FieldEnumeration(EnumerationDeclarator[FieldDeclarator]):
        pass

    assert FieldEnumeration.__member_types__ == (FieldDeclarator,)


def test_mixed_inheritance_filters_relevant_bases():
    class FieldDeclarator(AnnotatableDeclarator):
        __types__ = (Declarator,)

        def __validate_type__(self, type: Any) -> bool:  # noqa: A003
            return super().__validate_type__(type)

    class FieldEnumeration(EnumerationDeclarator[FieldDeclarator]):
        pass

    class Composite(FieldDeclarator, FieldEnumeration):
        pass

    assert Composite.__types__ == (Declarator,)
    assert Composite.__member_types__ == (FieldDeclarator,)


def test_conflicting_annotatable_types_raise():
    class IntDeclarator(AnnotatableDeclarator):
        __types__ = (int,)

        def __validate_type__(self, type: Any) -> bool:  # noqa: A003
            return super().__validate_type__(type)

    class StrDeclarator(AnnotatableDeclarator):
        __types__ = (str,)

        def __validate_type__(self, type: Any) -> bool:  # noqa: A003
            return super().__validate_type__(type)

    with pytest.raises(TypeError):
        class _Conflict(IntDeclarator, StrDeclarator):
            pass


def test_conflicting_enumeration_member_types_raise():
    class FieldDeclarator(AnnotatableDeclarator):
        __types__ = (Declarator,)

        def __validate_type__(self, type: Any) -> bool:  # noqa: A003
            return super().__validate_type__(type)

    class FirstEnumeration(EnumerationDeclarator[FieldDeclarator]):
        pass

    class OtherDeclarator(AnnotatableDeclarator):
        __types__ = (Declarator,)

        def __validate_type__(self, type: Any) -> bool:  # noqa: A003
            return super().__validate_type__(type)

    class SecondEnumeration(EnumerationDeclarator[OtherDeclarator]):
        pass

    with pytest.raises(TypeError):
        class _Conflict(FirstEnumeration, SecondEnumeration):
            pass
