#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from six.moves import mock


def test_valid_auth_code():
    from sasctl.core import AuthCodeAuth

    with mock.patch('sasctl.core.requests.post') as mock_post:
        with mock.patch('sasctl.core.input') as mock_input:

            # Don't prompt user for auth code
            mock_input.return_value = 'FooBar'

            # Don't actually call SASLogon service
            mock_post.return_value.json.return_value = {'token_type': 'bearer',
                                                        'access_token': '123',
                                                        'refresh_token': '456'}

            auth = AuthCodeAuth('https://example.com')

            assert mock_post.call_args[0][0] == 'https://example.com/SASLogon/oauth/token'
            assert mock_post.call_args[1]['data'] == {'code': 'FooBar', 'grant_type': 'authorization_code'}
            assert auth.access_token == '123'
            assert auth.refresh_token == '456'


