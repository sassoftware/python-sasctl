#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from sasctl.services import model_publish as mp

def test_publish_name():
    assert 'ModuleName' == mp._publish_name('Module Name') # Remove spaces
    assert '_1stModule' == mp._publish_name('1st Module')  # Cannot start with numbers
    assert 'ValidModule' == mp._publish_name('$&^Va*li#d   @Modu(le)!')

