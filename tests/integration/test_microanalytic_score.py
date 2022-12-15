#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl.services import microanalytic_score as mas

pytestmark = pytest.mark.usefixtures("session")


@pytest.mark.incremental
class TestMicroAnalyticScore:
    MODULE_NAME = "sasctl_testmodule"

    def test_create_python_module(self):
        source = "\n".join(
            (
                "def myfunction(var1, var2):",
                "    'Output: out1, out2'",
                "    out1 = var1 + 5",
                "    out2 = var2.upper()",
                "    return out1, out2",
                "def myfunction2(var1, var2):",
                "    'Output: out1'",
                "    return var1 + var2",
            )
        )

        r = mas.create_module(source=source, name=self.MODULE_NAME)
        assert self.MODULE_NAME == r.id
        assert "public" == r.scope

    def test_call_python_module_steps(self):
        r = mas.define_steps(self.MODULE_NAME)
        assert (6, "TEST") == r.myfunction(1, "test")

    def test_call_python_module_steps_pandas(self):
        pd = pytest.importorskip("pandas")

        r = mas.define_steps(self.MODULE_NAME)
        df = pd.DataFrame(dict(var1=[1], var2=["test"]))
        assert (6, "TEST") == r.myfunction(df.iloc[0, :])

        df = pd.DataFrame(dict(var1=[1.5], var2=[3]))
        assert r.myfunction2(df.iloc[0, :]) == 4.5

    def test_call_python_module_steps_numpy(self):
        np = pytest.importorskip("numpy")

        r = mas.define_steps(self.MODULE_NAME)
        array = np.array([1.5, 3])
        assert r.myfunction2(array) == 4.5

    def test_list_modules(self):
        all_modules = mas.list_modules()

        assert isinstance(all_modules, list)
        assert len(all_modules) > 0
        assert any(x.id == self.MODULE_NAME for x in all_modules)

    def test_get_module(self):
        module = mas.get_module(self.MODULE_NAME)

        assert module.id == self.MODULE_NAME

    def test_list_module_steps(self):
        steps = mas.list_module_steps(self.MODULE_NAME)

        assert isinstance(steps, list)
        assert any(s.id == "myfunction" for s in steps)

    def test_delete_module(self):
        assert mas.get_module(self.MODULE_NAME) is not None

        mas.delete_module(self.MODULE_NAME)

        assert mas.get_module(self.MODULE_NAME) is None
