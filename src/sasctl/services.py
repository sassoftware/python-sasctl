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
from ._services.microanalytic_score import MicroAnalyticScore as microanalytic_score
from ._services.model_management import ModelManagement as model_management
from ._services.model_publish import ModelPublish as model_publish
from ._services.model_repository import ModelRepository as model_repository
from ._services.projects import Projects as projects
from ._services.relationships import Relationships as relationships
from ._services.reports import Reports as reports
from ._services.report_images import ReportImages as report_images
from ._services.saslogon import SASLogon as saslogon
from ._services.sentiment_analysis import SentimentAnalysis as sentiment_analysis
from ._services.text_categorization import TextCategorization as text_categorization
from ._services.text_parsing import TextParsing as text_parsing
