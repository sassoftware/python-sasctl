#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock


def test_build_parser():
    """Verify arguments are correctly parsed."""
    from sasctl.utils.cli import _build_parser
    from sasctl.utils.cli import ArgInfo

    func = mock.MagicMock()
    func._cli_arguments.return_value = [ArgInfo("name", "str", True, None, "")]
    func.__doc__ = "docstring for a mock function."

    parser = _build_parser({"folders": {"delete": func}})
    args = parser.parse_args("folders delete myfolder".split())

    assert "folders" == args.service
    assert "delete" == args.command

    args = parser.parse_args("-vv -k folders delete myfolder".split())

    assert "folders" == args.service
    assert "delete" == args.command
    assert 2 == args.verbose
    assert args.insecure


def test_get_func_description():
    """Verify that function descriptions are correctly extracted from docstrings."""
    from sasctl.utils.cli import _get_func_description

    def func():
        """A function

        Some other info about the function

        Parameters
        ----------
        No parameters here

        Returns
        -------

        """

        assert "A function" == _get_func_description(func)


def test_service_names():
    from sasctl.utils.cli import _find_services

    services = _find_services()

    assert all(isinstance(service, str) for service in services)

    for service in ["folders", "models", "projects", "repositories"]:
        assert service in services


def test_decorator():
    from sasctl.utils.cli import ArgInfo
    from sasctl.utils.cli import sasctl_command

    @sasctl_command
    def command1(x):
        return x

    assert "testing" == command1("testing")
    assert "command1" == command1._cli_command
    assert command1._cli_service is None
    args = command1._cli_arguments()
    assert 1 == len(args)
    assert (
        ArgInfo(name="x", type="str", required=True, default=None, doc=None) == args[0]
    )

    @sasctl_command("mycmd")
    def command2(x):
        return x

    assert "testing" == command2("testing")
    assert "mycmd" == command2._cli_command
    assert command2._cli_service is None
    args = command2._cli_arguments()
    assert 1 == len(args)
    assert (
        ArgInfo(name="x", type="str", required=True, default=None, doc=None) == args[0]
    )

    @sasctl_command("myservice", "mycmd")
    def command3(x):
        return x

    assert "testing" == command3("testing")
    assert "mycmd" == command3._cli_command
    assert "myservice" == command3._cli_service
    args = command3._cli_arguments()
    assert 1 == len(args)
    assert (
        ArgInfo(name="x", type="str", required=True, default=None, doc=None) == args[0]
    )

    @sasctl_command
    def list_widgets(name, method="default"):
        """List the widgets

        Parameters
        ----------
        name : str
            name of the widget
        method : str
            parameter with a default

        Returns
        -------

        """
        result = method + str(name)
        return result

    assert "defaultwidget" == list_widgets("widget")
    assert "list" == list_widgets._cli_command
    assert "widgets" == list_widgets._cli_service
    args = list_widgets._cli_arguments()
    assert 2 == len(args)
    assert (
        ArgInfo(
            name="name",
            type="str",
            required=True,
            default=None,
            doc="name of the widget",
        )
        == args[0]
    )
    assert (
        ArgInfo(
            name="method",
            type="str",
            required=False,
            default="default",
            doc="parameter with a default",
        )
        == args[1]
    )
