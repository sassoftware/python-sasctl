#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__version__ = '1.0.1'
__author__ = 'SAS'
__credits__ = ['Yi Jian Ching, Lucas De Paula, Peter Tobac, Chris Toth, Jon '
               'Walker']
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright © 2019, SAS Institute Inc., ' \
                'Cary, NC, USA.  All Rights Reserved.'

import logging

from .core import Session, HTTPError, current_session
from .tasks import *


# Prevent package from emitting log records unless consuming
# application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())