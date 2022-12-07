#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl.utils.cli import main


pytestmark = pytest.mark.usefixtures("session")


def test_insecure_connection():
    """Verify that an insecure connection flag works."""

    r = main(["-k", "folders", "get", "dummy_folder"])
    assert r is None


def test_secure_connection():
    """Verify that a secure connection flag works."""

    r = main(["folders", "get", "dummy_folder"])
    assert r is None


def test_command_without_args(capsys):
    """Verify that a simple command works."""

    main(["folders", "list"])

    captured = capsys.readouterr()
    assert "Application Data" in captured.out
    assert "Model Repositories" in captured.out


def test_command_with_args(capsys):
    """Verify that a simple command with arguments works."""
    import json

    FOLDER_NAME = "Public"

    main(["folders", "get", FOLDER_NAME])

    captured = capsys.readouterr()

    # Output should be valid JSON
    output = json.loads(captured.out)

    assert output["name"] == FOLDER_NAME
    assert "id" in output
    assert "links" in output
