#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['files',
           'folders',
           'microanalytic_score',
           'model_management',
           'model_publish',
           'model_repository',
           'projects'
           ]

def _instantiate(name):
    module_name, class_name = name.rsplit('.', 1)
    module = __import__(module_name, fromlist=[''])
    cls = module.__dict__[class_name]
    return cls()

cas_management = _instantiate('sasctl._services.cas_management.CASManagement')
concepts = _instantiate('sasctl._services.concepts.Concepts')
data_sources = _instantiate('sasctl._services.data_sources.DataSources')
files = _instantiate('sasctl._services.files.Files')
folders = _instantiate('sasctl._services.folders.Folders')
microanalytic_score = _instantiate(
    'sasctl._services.microanalytic_store.MicroAnalyticScore')
model_management = _instantiate(
    'sasctl._services.model_management.ModelManagement')
model_publish = _instantiate(
    'sasctl._services.model_publish.ModelPublish')
model_repository = _instantiate(
    'sasctl._services.model_repository.ModelRepository')

