#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import textwrap
import warnings


class ExperimentalWarning(UserWarning):
    """Warning raised by @experimental decorator."""

    pass


def _insert_docstring_text(func, text):
    docstring = func.__doc__ or ""

    # Dedent the existing docstring.  Multi-line docstrings are only indented
    # after the first line, so split before dedenting.
    if "\n" in docstring:
        first_line, remainder = docstring.split("\n", 1)
        docstring = first_line + "\n" + textwrap.dedent(remainder)
    else:
        docstring = textwrap.dedent(docstring)

    docstring = docstring.strip()

    # Sphinx formatting requires 2 blank lines after a numpydoc section or one
    # blank line after another Sphinx directive.
    if docstring.split("\n")[-1].startswith(".. "):
        # Last line is another Sphinx directive
        gap = "\n\n"
    else:
        gap = "\n\n\n"

    return docstring + gap + text + "\n"


def deprecated(reason=None, version=None, removed_in=None):
    """Decorate a function or class to designated it as deprecated.

    Will raise a `DeprecationWarning` when used and automatically adds a
    Sphinx '.. deprecated::' directive to the docstring.

    Parameters
    ----------
    reason : str, optional
        User-friendly reason for deprecating.
    version : str
        Version in which initially marked as deprecated.
    removed_in : str, optional
        Version in which the class or function will be removed.

    Returns
    -------
    decorator

    """
    if version is None:
        raise ValueError("version must be specified.")

    def decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            warning = "%s is deprecated since version %s" % (func.__name__, version)

            if removed_in is not None:
                warning += " and will be removed in version %s." % removed_in
            else:
                warning += " and may be removed in a future version."

            if reason is not None:
                warning = warning + "  " + reason

            warnings.warn(warning, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Generate Sphinx deprecated directive
        directive = ".. deprecated:: %s" % version

        if reason is not None:
            directive += "\n  %s" % reason

        try:
            functools.update_wrapper(_wrapper, func)

            # Insert directive into original docstring
            _wrapper.__doc__ = _insert_docstring_text(func, directive)
        except AttributeError:
            # __doc__ is not writable in Py27
            pass

        return _wrapper

    return decorator


def experimental(func):
    """Decorate a function or class to designated it as experimental.

    Will raise an `ExperimentalWarning` when used and automatically adds a
    Sphinx '.. warning::' directive to the docstring.

    Parameters
    ----------
    func

    Returns
    -------
    func

    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        warning = (
            "%s is experimental and may be modified or removed without warning."
            % func.__name__
        )
        warnings.warn(warning, category=ExperimentalWarning, stacklevel=2)
        return func(*args, **kwargs)

    type_ = "class" if isinstance(func, type) else "method"
    directive = (
        ".. warning:: This %s is experimental and may be modified or removed without warning."
        % type_
    )

    try:
        # Insert directive into original docstring
        _wrapper.__doc__ = _insert_docstring_text(func, directive)
    except AttributeError:
        # __doc__ is not writable in Py27
        pass

    return _wrapper


def versionadded(reason=None, version=None):
    """Decorate a function or class to identify the version it was added.

    Automatically adds a Sphinx '.. versionadded::' directive to the docstring.

    Parameters
    ----------
    reason : str, optional
        User-friendly reason for deprecating.
    version : str
        Version in which initially marked as deprecated.

    Returns
    -------
    decorator


    .. versionadded:: 1.5

    """
    if version is None:
        raise ValueError("version must be specified.")

    def decorator(func):
        # Generate Sphinx deprecated directive
        directive = ".. versionadded:: %s" % version

        if reason is not None:
            directive += "\n  %s" % reason

        try:
            # Insert directive into original docstring
            func.__doc__ = _insert_docstring_text(func, directive)
        except AttributeError:
            # __doc__ is not writable in Py27
            pass

        return func

    return decorator


def versionchanged(reason=None, version=None):
    """Decorate a function or class to identify the version it was changed.

    Automatically adds a Sphinx '.. versionchanged::' directive to the docstring.

    Parameters
    ----------
    reason : str, optional
        User-friendly reason for deprecating.
    version : str
        Version in which initially marked as deprecated.

    Returns
    -------
    decorator

    .. versionadded:: 1.5

    """
    if version is None:
        raise ValueError("version must be specified.")

    def decorator(func):
        # Generate Sphinx deprecated directive
        directive = ".. versionchanged:: %s" % version

        if reason is not None:
            directive += "\n  %s" % reason

        try:
            # Insert directive into original docstring
            func.__doc__ = _insert_docstring_text(func, directive)
        except AttributeError:
            # __doc__ is not writable in Py27
            pass

        return func

    return decorator
