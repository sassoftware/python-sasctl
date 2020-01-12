#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


def _instantiate(name):
    # Instantiate and return the specified class without cluttering up the
    # module with numerous imports.
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
    'sasctl._services.microanalytic_score.MicroAnalyticScore')
model_management = _instantiate(
    'sasctl._services.model_management.ModelManagement')
model_publish = _instantiate(
    'sasctl._services.model_publish.ModelPublish')
model_repository = _instantiate(
    'sasctl._services.model_repository.ModelRepository')
projects = _instantiate('sasctl._services.projects.Projects')
relationships = _instantiate('sasctl._services.relationships.Relationships')
reports = _instantiate('sasctl._services.reports.Reports')
report_images = _instantiate('sasctl._services.report_images.ReportImages')
sentiment_analysis = _instantiate(
    'sasctl._services.sentiment_analysis.SentimentAnalysis')
text_categorization = _instantiate(
    'sasctl._services.text_categorization.TextCategorization')
text_parsing = _instantiate('sasctl._services.text_parsing.TextParsing')


