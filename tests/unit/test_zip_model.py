#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from contextlib import closing
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from sasctl.pzmm.zip_model import ZipModel as zm


def _create_sample_archive(prefix=None, suffix=None, is_viya_4=False):
    tmp_dir = tempfile.TemporaryDirectory()
    if suffix:
        for s in suffix:
            _ = tempfile.NamedTemporaryFile(delete=False, suffix=s, dir=tmp_dir.name)
    if prefix:
        _ = tempfile.NamedTemporaryFile(
            delete=False, prefix=prefix, suffix=".py", dir=tmp_dir.name
        )
    bytes_zip = zm.zip_files(tmp_dir.name, "Unit_Test_Model", is_viya4=is_viya_4)
    # Check that for files with a valid extension, the generated zip file contains
    # the expected number of files
    with closing(ZipFile(Path(tmp_dir.name) / "Unit_Test_Model.zip")) as archive:
        num_files = len(archive.infolist())
    return bytes_zip, num_files


def test_zip_files():
    """
    Test cases:
    - Creates in memory zip
    - Writes zip to disk
    - Returns proper BytesIO object in both cases
    """
    model_files = {
        "Test.json": json.dumps({"Test": True, "TestNum": 1}),
        "Test.py": f"import sasctl\ndef score():\n{'':4}return \"Test score\"",
    }
    bytes_zip = zm.zip_files(model_files, "Unit_Test_Model")
    assert issubclass(BytesIO, type(bytes_zip))

    bytes_zip, _ = _create_sample_archive(suffix=[".json"])
    assert issubclass(BytesIO, type(bytes_zip))


def test_zip_files_filter():
    """
    Test cases:
    - Zip proper number of files for Viya 4 model
    - Zip proper number of files for Viya 3.5 model
    - Raise error if no valid files
    """
    suffix = [".json", ".pickle", ".mojo", ".txt"]
    prefix = "score_"

    # Viya 4 model
    _, num_files = _create_sample_archive(prefix, suffix, True)
    assert num_files == 4

    # Viya 3.5 model
    _, num_files = _create_sample_archive(prefix, suffix, False)
    assert num_files == 3

    # No valid files
    with pytest.raises(FileNotFoundError):
        _, _ = _create_sample_archive(suffix=[".txt"])
