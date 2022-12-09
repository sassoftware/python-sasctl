#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import difflib
import logging
import lzma
import os
import pickle
import re

from betamax import util, BaseMatcher, BaseSerializer
from betamax.matchers.path import PathMatcher
from urllib.parse import urlsplit


log = logging.getLogger("sasctl.betamax")


class BinarySerializer(BaseSerializer):
    """Serializes Betamax cassettes as compressed binary data.

    Notes
    -----
    As of Betamax 0.8.1 it is not possible to serialize binary data directly as BaseSerializer expects
    a string.  Betamax 0.9 will supposedly add this functionality, but it is unreleased as of 2022-11-28.

    As as workaround, we encode the compressed data as a string.  This adds some bloat but it's more than offset
    by the compression.

    """

    name = "binary"
    stored_as_binary = False

    @staticmethod
    def generate_cassette_name(cassette_library_dir, cassette_name):
        return os.path.join(cassette_library_dir, f"{cassette_name}.lzma")

    def serialize(self, cassette_data):
        data = pickle.dumps(cassette_data)
        data = lzma.compress(data)
        return base64.b85encode(data).decode("utf-8")

    def deserialize(self, cassette_data):
        try:
            deserialized_data = base64.b85decode(cassette_data)
            deserialized_data = lzma.decompress(deserialized_data)
            deserialized_data = pickle.loads(deserialized_data)
        except lzma.LZMAError:
            deserialized_data = {}

        return deserialized_data


class RedactedPathMatcher(PathMatcher):
    """Matches requests identically to the default behavior except that CAS s
    ession ids in the path are ignored.
    """

    name = "redacted_path"

    def _strip_session_id(self, path):
        match = re.search(r"(?<=sessions/)[0-9a-f\-]*", path)
        if match:
            path = path.replace(match.group(0), "")
        return path

    def match(self, request, recorded_request):
        request_path = self._strip_session_id(urlsplit(request.url).path)
        recorded_path = self._strip_session_id(urlsplit(recorded_request["uri"]).path)
        return request_path == recorded_path


class PartialBodyMatcher(BaseMatcher):
    # Matches based on the body of the request
    name = "partial_body"

    def match(self, request, recorded_request):
        recorded_request = util.deserialize_prepared_request(recorded_request)

        request_body = b""
        if request.body:
            request_body = util.coerce_content(request.body)

        recorded_body = b""
        if recorded_request.body:
            recorded_body = util.coerce_content(recorded_request.body)

        if recorded_body != request_body:
            diff = difflib.context_diff(recorded_body, request_body)
            log.debug("** Cassette Differences: **\n" + "\n".join(diff))

        return recorded_body == request_body
