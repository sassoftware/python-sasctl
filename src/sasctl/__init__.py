#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__version__ = '1.6.0'
__author__ = 'SAS'
__credits__ = ['Yi Jian Ching, Lucas De Paula, James Kochuba, Peter Tobac, '
               'Chris Toth, Jon Walker']
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright © 2019, SAS Institute Inc., ' \
                'Cary, NC, USA.  All Rights Reserved.'

import logging


from .core import current_session, delete, get, get_link, platform_version, post, put, request_link
from .core import RestObj, Session, HTTPError
from .tasks import register_model, publish_model, update_model_performance


# Prevent package from emitting log records unless consuming
# application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
