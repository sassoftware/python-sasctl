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
    pass

from .._services.model_repository import ModelRepository
from .._services.cas_management import CASManagement
from .._services.concepts import Concepts
from .._services.data_sources import DataSources
from .._services.files import Files
from .._services.folders import Folders

model_repository = ModelRepository()
cas_management = CASManagement()
concepts = Concepts()
data_sources = DataSources()
files = Files()
folders = Folders()

del Folders