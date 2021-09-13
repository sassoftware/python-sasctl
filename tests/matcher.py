#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from urllib.parse import urlsplit

from betamax.matchers.path import PathMatcher

class RedactedPathMatcher(PathMatcher):
    """Matches requests identically to the default behavior except that CAS s
    ession ids in the path are ignored.
    """

    name = 'redacted_path'

    def _strip_session_id(self, path):
        match = re.search(r'(?<=sessions/)[0-9a-f\-]*', path)
        if match:
            path = path.replace(match.group(0), '')
        return path

    def match(self, request, recorded_request):
        request_path = self._strip_session_id(urlsplit(request.url).path)
        recorded_path = self._strip_session_id(urlsplit(recorded_request['uri']).path)
        return request_path == recorded_path



from betamax import util, BaseMatcher
import difflib
import logging
log = logging.getLogger('sasctl.betamax')


class PartialBodyMatcher(BaseMatcher):
    # Matches based on the body of the request
    name = 'partial_body'

    def match(self, request, recorded_request):
        recorded_request = util.deserialize_prepared_request(recorded_request)

        request_body = b''
        if request.body:
            request_body = util.coerce_content(request.body)

        recorded_body = b''
        if recorded_request.body:
            recorded_body = util.coerce_content(recorded_request.body)

        if recorded_body != request_body:
            diff = difflib.context_diff(recorded_body, request_body)
            log.debug('** Cassette Differences: **\n' + '\n'.join(diff))

        return recorded_body == request_body