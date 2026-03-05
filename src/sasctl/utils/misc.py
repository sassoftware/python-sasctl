#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import string
import warnings

from .decorators import versionadded

# Mapping of Python import names to their PyPI installation names
IMPORT_TO_INSTALL_MAPPING = {
    # Data Science & ML Core
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    # Data Formats & Parsing
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "docx": "python-docx",
    "pptx": "python-pptx",
    # Date & Time Utilities
    "dateutil": "python-dateutil",
    # Database Connectors
    "MySQLdb": "MySQL-python",
    "psycopg2": "psycopg2-binary",
    # System & Platform
    "win32api": "pywin32",
    "win32com": "pywin32",
    # Scientific Libraries
    "Bio": "biopython",
}


def installed_packages():
    """List Python packages installed in the current environment.

    Returns
    -------

    Notes
    -----
    Uses pip freeze functionality, so pip module must be present. For pip
    versions >=20.1, this functionality fails to provide versions for some
    conda installed, locally installed, and url installed packages. Instead,
    uses the importlib package, which is typically bundled with python.

    """
    from packaging import version

    try:
        import pip

        if version.parse(pip.__version__) >= version.parse("20.1"):
            from importlib.metadata import distributions

            output = [p.name + "==" + p.version for p in distributions()]
            return output
        else:
            from pip._internal.operations import freeze
    except ImportError:

        try:
            from pip.operations import freeze
        except ImportError:
            freeze = None

    if freeze is not None:
        return list(freeze.freeze())


@versionadded(version="1.5.1")
def random_string(length):
    """Generates a random alphanumeric string of a given length.

    Parameters
    ----------
    length : int
        The length of the generate string.

    Returns
    -------
    str

    """

    # random.choices() was not added until Python 3.6, so repeatedly call .choice() instead
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


@versionadded(version="1.9.0")
def check_if_jupyter() -> bool:
    """
    Check if the code is being executed from a Jupyter notebook.

    Source: https://stackoverflow.com/questions/47211324/check-if-module-is-running-in-
    jupyter-or-not

    Returns
    -------
    bool
        True if a Jupyter notebook is detected. False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except (ImportError, NameError):
        return False
