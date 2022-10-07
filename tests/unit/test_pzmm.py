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


def _create_sample_archive(suffix, is_viya_4=False):
    from sasctl.pzmm.zip_model import ZipModel as zm
    tmp_dir = tempfile.TemporaryDirectory()
    for s in suffix:
        _ = tempfile.NamedTemporaryFile(delete=False, suffix=s, dir=tmp_dir.name)
    bytes_zip = zm.zip_files(tmp_dir.name, "Unit_Test_Model", is_viya4=is_viya_4)
    # Check that for files with a valid extension, the generated zip file contains the expected number of files
    with closing(ZipFile(Path(tmp_dir.name) / "Unit_Test_Model.zip")) as archive:
        num_files = len(archive.infolist())
    # (Path(tmp_dir.name) / "Unit_Test_Model.zip").unlink()
    return bytes_zip, num_files


def test_zip_files_return():
    """
    Unit test for the zip_files function in pzmm.zip_model. Verifies that a BytesIO object is returned.
    """
    bytes_zip, _ = _create_sample_archive([".json"])
    assert issubclass(BytesIO, type(bytes_zip))


def test_zip_files_filter():
    """
    Unit test for _filter_files function in pzmm.zip_model. Verifies that proper file lists are returned and an error is
    thrown if no valid files could be found.
    """
    suffix = [".json", ".pickle", ".mojo", "Score.py", ".txt"]

    # Viya 4 model
    _, num_files = _create_sample_archive(suffix, True)
    assert num_files == 4

    # Viya 3.5 model
    _, num_files = _create_sample_archive(suffix, False)
    assert num_files == 3

    # No valid files
    with pytest.raises(FileNotFoundError):
        _, _ = _create_sample_archive([".txt"])

