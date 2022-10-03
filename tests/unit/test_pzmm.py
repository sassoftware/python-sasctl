#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
import tempfile
from io import BytesIO
from contextlib import closing
from zipfile import ZipFile
from pathlib import Path


def test_zip_files():
    """
    Unit test for the zipFiles function in pzmm.zipModel. Verifies that globbing is working as expected and that a
    BytesIO object is returned. Also checks that for models used for SAS Viya 3.5 python score code and other
    unexpected extensions are not included in the zip file.
    """
    def create_test_archive(s, is_viya_4, flag):
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=s, dir=tmp_dir.name)
        # Verify that the zipFiles output is in a BytesIO format
        assert issubclass(BytesIO, type(zm.zipFiles(tmp_dir.name, "Unit_Test_Model", isViya4=is_viya_4)))
        # Check that for files with a valid extension, the generated zip file contains the expected number of files
        with closing(ZipFile(Path(tmp_dir.name) / "Unit_Test_Model.zip")) as archive:
            assert len(archive.infolist()) == flag
        (Path(tmp_dir.name) / "Unit_Test_Model.zip").unlink()

    from sasctl.pzmm.zipModel import ZipModel as zm

    # Verify that the expected extensions are collected correctly in the generated zip file
    for suffix in [".json", ".pickle", ".mojo", "Score.py"]:
        create_test_archive(suffix, True, True)

    # Verify that unexpected extensions are not collected in the generated zip file
    for suffix in [".txt", "Score.py"]:
        create_test_archive(suffix, False, False)
