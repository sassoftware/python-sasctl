#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


def test_publish_name():
    from sasctl.services.model_publish import _publish_name

    assert 'ModuleName' == _publish_name('Module Name') # Remove spaces
    assert '_1stModule' == _publish_name('1st Module')  # Cannot start with numbers
    assert 'ValidModule' == _publish_name('$&^Va*li#d   @Modu(le)!')

