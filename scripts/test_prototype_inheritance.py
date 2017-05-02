#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module description.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

from anaximander.meta import prototype, metacharacter, NxObject, typeinitmethod


registry = dict()


@prototype
class C(NxObject):
    c = metacharacter()

    @typeinitmethod
    def local_registration(cls):
        registry[cls.metacharacters] = cls


@prototype
class D(C, c='c'):
    d = metacharacter()


E = D.subtype(name='D', d='d')

print(globals())
