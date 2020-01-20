#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sasctl.services import microanalytic_score
from sasctl.services import model_repository as mr
from sasctl.tasks import register_model, publish_model
from sasctl.core import PagedItemIterator, RestObj

# Every test function in the module will automatically receive the session fixture
pytestmark = pytest.mark.usefixtures('session')


@pytest.mark.incremental
class TestAStoreModel:
    project_name = 'SASCTL AStore Test Project'
    model_name = 'AStore Model'

    def test_model_import(self, astore):
        """Import a model from an ASTORE"""
        model = register_model(astore, self.model_name, self.project_name, force=True)

        assert self.model_name == model.name

    def test_get_model_contents(self):
        # Resolves https://github.com/sassoftware/python-sasctl/issues/33
        content = mr.get_model_contents(self.model_name)

        assert isinstance(content, list)
        assert 'AstoreMetadata.json' in [str(c) for c in content]
        assert 'ModelProperties.json' in [str(c) for c in content]
        assert 'inputVar.json' in [str(c) for c in content]
        assert 'outputVar.json' in [str(c) for c in content]

    def test_create_model_version(self):
        r = mr.create_model_version(self.model_name)
        assert r.modelVersionName == '2.0'

        r = mr.create_model_version(self.model_name, minor=True)
        assert r.modelVersionName == '2.1'

    def test_get_model_versions(self):
        versions = mr.list_model_versions(self.model_name)
        assert isinstance(versions, list)

        # NOTE: the current version (2.1) is NOT included
        assert len(versions) == 2
        assert any(v.modelVersionName == '1.0' for v in versions)
        assert any(v.modelVersionName == '2.0' for v in versions)

    def test_copy_astore(self):
        """Copy the ASTORE to filesystem for MAS"""
        job = mr.copy_analytic_store(self.model_name)

        assert job.state == 'pending'

    def test_model_publish(self):
        """Publish the imported model to MAS"""

        pytest.xfail('Import job pends forever.  Waiting on Support Track #7612766673')
        module = publish_model(self.model_name, 'maslocal', max_retries=60)
        self.module_name = module.name

        print('done')

    def test_module_execute(self):
        response = microanalytic_score.execute_module_step(self.module_name, 'score', Income=1e4, Credit_Limit=4000)
        print('done')

    def test_delete_model(self):
        num_models = len(mr.list_models())
        model = mr.get_model(self.model_name)
        mr.delete_model(model)

        model = mr.get_model(model.id)
        assert model is None

        all_models = mr.list_models()
        assert len(all_models) == num_models - 1

    def test_delete_project(self):
        num_projects = len(mr.list_projects())
        project = mr.get_project(self.project_name)
        mr.delete_project(project)

        project = mr.get_project(self.project_name)
        assert project is None

        all_projects = mr.list_projects()
        assert len(all_projects) == num_projects - 1



@pytest.mark.incremental
class TestBasicModel:
    project_name = 'SASCTL Basic Test Project'
    model_name = 'Basic Model'

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
        repo = mr.default_repository().get('id')

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
