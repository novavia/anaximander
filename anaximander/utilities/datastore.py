#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides base functionalities to manage data storage resources.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
import warnings

# =============================================================================
# Storage resource base class.
# =============================================================================


class ResourceError(Exception):
    """Customize exception for administrative resource errors."""
    pass


class StorageResource(abc.ABC):
    """A Mixin class that provides basic administrative functions."""

    @abc.abstractmethod
    def __exists__(self):
        """Tests existence of the resource. Must return True or False."""
        return False

    @abc.abstractmethod
    def __empty__(self):
        """Tests whether the resource is empty or not."""
        return True

    @abc.abstractmethod
    def __create__(self, **kwargs):
        """Type-specific creation method."""
        pass

    @abc.abstractmethod
    def __drop__(self, force=False, **kwargs):
        """Type-specific drop method."""
        pass

    def warn(self, msg):
        warnings.warn(msg, ResourceWarning)

    @property
    def exists(self):
        return self.__exists__()

    @property
    def empty(self):
        if not self.__exists__():
            return True
        return self.__empty__()

    def create(self, overwrite=False, warn=True, force=False, **kwargs):
        """Creation method.

        Params:
            overwrite: unless True, the method systematically fails in
                case the resource already exists.
            warn: whether to provide a warning and user input before
                overwriting an existing resource.
            force: passed on to drop if a call is required.
            **kwargs: any kwargs accepted by __create__.
        """
        if self.exists:
            if warn:
                msg = "Attempting to create {0}, which already exists."
                if overwrite is True:
                    self.drop(force=force)
                else:
                    msg += " Aborting create method."
                    self.warn(msg.format(self))
                    return False
            else:
                if overwrite is True:
                    self.drop(confirm=False, force=force)
                else:
                    return False
        self.__create__(**kwargs)
        return True

    def drop(self, confirm=True, force=False, **kwargs):
        """Resource deletion method.

        Params:
            confirm: unless False, a user prompt is provided to avoid
                accidental deletions. Otherwise, deletion proceeds.
            force: unless True, the method only succeeds if the resource
                is already empty.
            **kwargs: any kwargs accepted by __drop__.
        """
        if not self.exists:
            msg = "{0} doesn't exist."
            raise ResourceError(msg.format(self))
        if force is not True and not self.empty:
            msg = "Cannot drop non-empty {0}."
            self.warn(msg.format(self))
            return False
        if confirm is not False:
            msg = "This will permanently delete {0}. Are you sure you " + \
                  "would like to proceed? (Y/n)."
            confirmation = input(msg.format(self))
            if not confirmation == 'Y':
                msg = "Aborting drop method."
                print(msg)
                return False
        self.__drop__(force=True, **kwargs)
        return True
