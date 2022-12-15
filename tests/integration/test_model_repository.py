#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl.core import RestObj
from sasctl.services import microanalytic_score
from sasctl.services import model_repository as mr
from sasctl.tasks import register_model, publish_model

# Every test function in the module will automatically receive the session fixture
pytestmark = pytest.mark.usefixtures("session")


@pytest.mark.incremental
class TestAStoreModel:
    PROJECT_NAME = "sasctl_testing TestAstoreModel"
    MODEL_NAME = "IrisGradBoost"

    def test_model_import(self, iris_astore):
        """Import a model from an ASTORE"""
        model = register_model(
            iris_astore, self.MODEL_NAME, self.PROJECT_NAME, force=True
        )

        assert self.MODEL_NAME == model.name

    def test_get_model_contents(self):
        # Resolves https://github.com/sassoftware/python-sasctl/issues/33
        content = mr.get_model_contents(self.MODEL_NAME)

        assert isinstance(content, list)

    def test_create_model_version(self):
        r = mr.create_model_version(self.MODEL_NAME)
        assert r.modelVersionName == "2.0"

        r = mr.create_model_version(self.MODEL_NAME, minor=True)
        assert r.modelVersionName == "2.1"

    def test_get_model_versions(self):
        versions = mr.list_model_versions(self.MODEL_NAME)
        assert isinstance(versions, list)

        # NOTE: the current version (2.1) is NOT included
        assert len(versions) == 2
        assert any(v.modelVersionName == "1.0" for v in versions)
        assert any(v.modelVersionName == "2.0" for v in versions)

    def test_copy_astore(self):
        """Copy the ASTORE to filesystem for MAS"""
        job = mr.copy_analytic_store(self.MODEL_NAME)

        assert job.state == "pending"

    def test_model_publish(self, cache):
        """Publish the imported model to MAS"""

        module = publish_model(
            self.MODEL_NAME, "maslocal", max_retries=60, replace=True
        )

        # SAS module should have a "score" method
        assert "score" in module.stepIds

        # Store module name so we can retrieve it in later tests
        cache.set("MAS_MODULE_NAME", module.name)

    def test_module_execute(self, cache, iris_dataset):
        # Store module name so we can retrieve it in later tests
        module_name = cache.get("MAS_MODULE_NAME", None)

        x = iris_dataset.drop("Species", axis=1).iloc[0, :]

        response = microanalytic_score.execute_module_step(
            module_name, "score", **x.to_dict()
        )

        assert all(
            k in response
            for k in ("P_Species0", "P_Species1", "P_Species2", "I_Species")
        )

    def test_delete_model(self):
        num_models = len(mr.list_models())
        model = mr.get_model(self.MODEL_NAME)
        mr.delete_model(model)

        model = mr.get_model(model.id)
        assert model is None

        all_models = mr.list_models()
        assert len(all_models) == num_models - 1

    def test_delete_project(self):
        num_projects = len(mr.list_projects())
        project = mr.get_project(self.PROJECT_NAME)
        mr.delete_project(project)

        project = mr.get_project(self.PROJECT_NAME)
        assert project is None

        all_projects = mr.list_projects()
        assert len(all_projects) == num_projects - 1


@pytest.mark.incremental
class TestBasicModel:
    project_name = "SASCTL Basic Test Project"
    model_name = "Basic Model"

    def test_project_missing(self):
        # Creating a model in a non-existent project should fail
        with pytest.raises(ValueError):
            mr.create_model(self.model_name, self.project_name)

    def test_create_project(self):
        # Count the current # of projects in the environment
        projects = mr.list_projects()
        assert isinstance(projects, list)
        project_count = len(projects)

        # Create a new project
        repo = mr.default_repository().get("id")

        project = mr.create_project(self.project_name, repo)
        assert isinstance(project, RestObj)

        # Total number of projects should have increased
        projects = mr.list_projects()
        assert len(projects) == project_count + 1

    def test_create_model(self):
        # Count the current # of models in the environment
        models = mr.list_models()
        assert isinstance(models, list)
        model_count = len(models)

        # Create a new model
        model = mr.create_model(self.model_name, self.project_name)
        assert isinstance(model, RestObj)
        assert model.name == self.model_name

        # Total number of models should have increased
        models = mr.list_models()
        assert len(models) == model_count + 1

    def test_delete_model(self):
        model_count = len(mr.list_models())
        model1 = mr.get_model(self.model_name)
        assert isinstance(model1, RestObj)

        model2 = mr.get_model(model1.id)
        assert model1.id == model2.id

        mr.delete_model(model1)

        model3 = mr.get_model(model1.id)
        assert model3 is None

        all_models = mr.list_models()
        assert len(all_models) == model_count - 1

    def test_delete_project(self):
        project_count = len(mr.list_projects())
        project = mr.get_project(self.project_name)
        assert isinstance(project, RestObj)

        mr.delete_project(project)

        project = mr.get_project(self.project_name)
        assert project is None

        all_projects = mr.list_projects()
        assert len(all_projects) == project_count - 1
