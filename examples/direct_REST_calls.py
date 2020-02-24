#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle                                                 # skipcq: BAN-B403
from sasctl import get, get_link, request_link, Session

# Authenticate to the SAS environment
s = Session('example.sas.com', 'brian', 'N@ughtiusMax1mus')

# Use the established session to GET example.sas.com/files (the root URI for
# the Files service)
response = get('files')

# List all of the links currently available from the root URL
for link in response.links:
    print(link)

# Retrieve info about the "files" link, including URL, content type, etc.
# Only returns information about the link, does not actually make a REST call.
link = get_link(response, 'files')

# Actually make the request identified by the "files" link.  In this example,
# that corresponds to GET example.sas.com/files/files which returns a paginated
# list of all files available.
all_files = request_link(response, 'files')

# Iterate through all files and print those with a particular filename.
# NOTE: this is NOT a recommended approach as the client must retrieve all
# filenames from the server to filter.
for file in filter(lambda x: x.name == 'traincode.sas', all_files):
    print(file)

# Make the same request as before, but with a service-side filter to perform
# the filtering.
all_files = request_link(response, 'files', params={'filter': 'eq(name, "traincode.sas")'})

# Select the first file matching the filter criteria.
# NOTE: this is not the actual file, just a collection of metadata about the
# file and the associated REST links available.
file = all_files[0]

# Make a request to the "content" link to retrieve the actual file contents.
content = request_link(file, 'content')
print(content)

# Make a request for files where the filename matches "model.pkl"
# NOTE: this example assumes there is a single matching file on the server.  If
# there are multiple such files you will need to select one.
file = request_link(response, 'files', params={'filter': 'eq(name, "model.pkl")'})
if file:
    file = file[0]

# Request the contents of the file.
# NOTE: because the file is a binary pickle file, the "format='content'"
# parameter is required to indicate that we want to raw content of the response
# without attempting to parse it as text or JSON.
pkl = request_link(file, 'content', format='content')

# Load the pickled file to reconstitute the Python object on the client.
# WARNING: you should not unpickle objects from untrusted sources as they may
# contain malicious content.  Additionally, unpickling is likely to fail if
# the Python environment on the client differs from the environment where the
# object was first created.
pickle.loads(pkl)                                             # skipcq: BAN-B301
