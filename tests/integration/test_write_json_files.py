#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2023, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from sasctl.pzmm.write_json_files import JSONFiles as jf

pytestmark = pytest.mark.usefixtures("session")


def _classification_model(data, target):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    data = pd.get_dummies(data, drop_first=True).fillna(data.mean())
    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(columns=target), data[target], test_size=0.3
    )

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    return model, x_test, y_test


def test_calculate_model_statistics(cas_session, hmeq_dataset):
    """
    Test Cases:
    - No output files
    - Output files for fitstat, roc, and lift
    """
    model, x, y = _classification_model(hmeq_dataset, "BAD")

    predict = model.predict(x)
    proba = model.predict_proba(x)
    predict_proba = []
    for i, row in enumerate(proba):
        predict_proba.append(row[int(predict[i])])
    predict_df = pd.DataFrame(
        {"predict": list(predict), "proba": predict_proba}, index=y.index
    )
    test_data = pd.concat([y, predict_df], axis=1)

    json_dicts = jf.calculate_model_statistics(target_value="1", test_data=test_data)
    assert "dmcas_fitstat.json" in json_dicts
    assert "dmcas_roc.json" in json_dicts
    assert "dmcas_lift.json" in json_dicts

    with tempfile.TemporaryDirectory() as tmp_dir:
        jf.calculate_model_statistics(
            target_value="1", test_data=test_data, json_path=Path(tmp_dir)
        )
        assert (Path(tmp_dir) / "dmcas_fitstat.json").exists()
        assert (Path(tmp_dir) / "dmcas_roc.json").exists()
        assert (Path(tmp_dir) / "dmcas_lift.json").exists()
