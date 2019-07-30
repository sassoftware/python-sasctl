#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re

from setuptools import setup, find_packages


def read_dunder(name):
    with open(os.path.join('src', 'sasctl', '__init__.py')) as f:
        for line in f.readlines():
            match = re.search(r'(?<=^__{}__ = [\'"]).*\b'.format(name), line)
            if match:
                return match.group(0)
        raise RuntimeError('Unable to find __%s__ in __init__.py' % name)


def get_file(filename):
    # print('abspath: %s' % os.path.abspath(__file__))
    # print('dirname: %s' % os.path.dirname(os.path.abspath(__file__)))
    # folder = os.path.abspath(os.path.dirname(__file__))
    # print('joined: %s' % os.path.join(folder, filename))
    # for file in os.listdir(folder):
    #     print('file: %s' % file)
    #
    # filename = os.path.join(folder, filename)
    # print('filename: %s' % filename)

    # assert False
    with open(filename, 'r') as f:
        return f.read()


setup(
    name='sasctl',
    include_package_data=True,
    license='Apache v2.0',
    description='SAS Viya REST Client',
    long_description=get_file('README.md'),
    long_description_content_type='text/markdown',
    version=read_dunder('version'),
    author=read_dunder('author'),
    url='https://github.com/sassoftware/python-sasctl/',
    project_urls={
        'Bug Tracker': 'https://github.com/sassoftware/python-sasctl/issues',
        'Documentation': 'https://sassoftware.github.io/python-sasctl/',
        'Source Code': 'https://github.com/sassoftware/python-sasctl'
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires='>=2.7',
    install_requires=[
        'requests',
        'six >= 1.11'
    ],
    extras_require = {
        'swat': ['swat'],
        'kerberos': ['kerberos ; platform_system != "Windows"',
                     'winkerberos ; platform_system == "Windows"'],
        'all': ['swat',
                'kerberos ; platform_system != "Windows"',
                'winkerberos ; platform_system == "Windows"'],
    },
    entry_points = {'console_scripts': ['sasctl = sasctl.utils.cli:main']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent'
    ]
)