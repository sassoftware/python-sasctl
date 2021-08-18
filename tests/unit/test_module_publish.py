#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

from sasctl.services import model_publish as mp


def test_publish_name():
    assert 'ModuleName' == mp._publish_name('Module Name')  # Remove spaces
    assert '_1stModule' == mp._publish_name('1st Module')  # Cannot start with numbers
    assert 'ValidModule' == mp._publish_name('$&^Va*li#d   @Modu(le)!')


def test_create_cas_destination():
    target = {
        'name': 'caslocal',
        'destinationType': 'cas',
        'casServerName': 'camelot',
        'casLibrary': 'round',
        'destinationTable': 'table',
        'description': None,
    }

    with mock.patch('sasctl._services.model_publish.ModelPublish.post') as post:
        mp.create_cas_destination(
            'caslocal', server='camelot', library='round', table='table'
        )

        assert post.called
        json = post.call_args[1]['json']

        for k in json.keys():
            assert json[k] == target[k]


def test_create_mas_destination():
    target = {
        'name': 'spam',
        'destinationType': 'microAnalyticService',
        'masUri': 'http://spam.com',
        'description': 'Real-time spam',
    }

    with mock.patch('sasctl._services.model_publish.ModelPublish.post') as post:
        mp.create_mas_destination(
            target['name'], target['masUri'], target['description']
        )

        assert post.called
        json = post.call_args[1]['json']

        for k in json.keys():
            assert json[k] == target[k]
