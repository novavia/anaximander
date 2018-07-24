#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to Anaximander's transforms package.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
