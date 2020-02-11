#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Import all of the services from their respective modules to provide
# a single location for importing services.  All services should utilize
# classmethods allowing them to be used without instantiation
from ._services.cas_management import CASManagement as cas_management
from ._services.concepts import Concepts as concepts
from ._services.data_sources import DataSources as data_sources
from ._services.files import Files as files
from ._services.folders import Folders as folders
from ._services.microanalytic_score import \
    MicroAnalyticScore as microanalytic_score
from ._services.ml_pipeline_automation import \
    MLPipelineAutomation as ml_pipeline_automation
from ._services.model_management import ModelManagement as model_management
from ._services.model_publish import ModelPublish as model_publish
from ._services.model_repository import ModelRepository as model_repository
from ._services.projects import Projects as projects
from ._services.relationships import Relationships as relationships
from ._services.reports import Reports as reports
from ._services.report_images import ReportImages as report_images
from ._services.sentiment_analysis import \
    SentimentAnalysis as sentiment_analysis
from ._services.text_categorization import \
    TextCategorization as text_categorization
from ._services.text_parsing import TextParsing as text_parsing

# def _instantiate(name):
#     # Instantiate and return the specified class without cluttering up the
#     # module with numerous imports.
#     module_name, class_name = name.rsplit('.', 1)
#     module = __import__(module_name, fromlist=[''])
#     cls = module.__dict__[class_name]
#     return cls()

# cas_management = _instantiate('sasctl._services.cas_management.CASManagement')
# concepts = _instantiate('sasctl._services.concepts.Concepts')
# data_sources = _instantiate('sasctl._services.data_sources.DataSources')
# files = _instantiate('sasctl._services.files.Files')
# folders = _instantiate('sasctl._services.folders.Folders')
# microanalytic_score = _instantiate(
#     'sasctl._services.microanalytic_score.MicroAnalyticScore')
#
# ml_pipeline_automation = _instantiate(
#     'sasctl._services.ml_pipeline_automation.MLPipelineAutomation')
# model_management = _instantiate(
#     'sasctl._services.model_management.ModelManagement')
# model_publish = _instantiate(
#     'sasctl._services.model_publish.ModelPublish')
# model_repository = _instantiate(
#     'sasctl._services.model_repository.ModelRepository')
# projects = _instantiate('sasctl._services.projects.Projects')
# relationships = _instantiate('sasctl._services.relationships.Relationships')
# reports = _instantiate('sasctl._services.reports.Reports')
# report_images = _instantiate('sasctl._services.report_images.ReportImages')
# sentiment_analysis = _instantiate(
#     'sasctl._services.sentiment_analysis.SentimentAnalysis')
# text_categorization = _instantiate(
#     'sasctl._services.text_categorization.TextCategorization')
# text_parsing = _instantiate('sasctl._services.text_parsing.TextParsing')
