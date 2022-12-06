#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import inspect
import json
import logging
import os
import pkgutil
import warnings
from collections import namedtuple, defaultdict
from importlib import import_module
from pprint import pprint


ArgInfo = namedtuple("ArgInfo", ["name", "type", "required", "default", "doc"])


def sasctl_command(name, subname=None):
    """Decorator that tags the function as being usable from the command line.

    Parameters
    ----------
    name : str
        the name of the command that will be shown on the command line.
    subname : str
        the name of the service that the command will be listed under

    Returns
    -------
    function

    Examples
    --------
    Define a command called 'cmd' not associated with a service
    >>> @sasctl_command('cmd')
    >>> def func():
            ...

    Define a command called 'cmd' associated with the 'svc' service
    >>> @sasctl_command('svc', 'cmd')
    >>> def func():
            ...

    Define a command and allow it's name and service to be auto-assigned
    >>> @sasctl_command
    >>> def func():
            ...

    """

    def decorator(func):
        if isinstance(name, str):
            if isinstance(subname, str):
                command_name = subname
                service_name = name
            else:
                command_name = name
                service_name = subname
        else:
            command_name = func.__name__
            if any(
                command_name.startswith(x)
                for x in ["list_", "update_", "get_", "create_", "delete_"]
            ):
                parts = command_name.split("_")
                command_name = parts[0]
                service_name = parts[-1]
            else:
                service_name = subname

        def parse_args():
            """Retrieve argument metadata from function signature and docstring."""
            arg_spec = inspect.getfullargspec(func)
            defaults = list(arg_spec.defaults) if arg_spec.defaults is not None else []
            required = [True] * (len(arg_spec.args) - len(defaults)) + [False] * len(
                defaults
            )
            defaults = [None] * (len(arg_spec.args) - len(defaults)) + defaults
            types = []
            help_doc = []

            doc = inspect.getdoc(func)
            if doc and doc.find("Parameters\n"):
                doc_lines = doc[doc.find("Parameters\n") :].splitlines()
                doc_lines.pop(0)  # First line is "Parameters"

                if doc_lines and doc_lines[0].startswith("---"):
                    doc_lines.pop(
                        0
                    )  # Discard ----------- line under "Parameters" heading

                while doc_lines:
                    var = doc_lines.pop(0)

                    if var.startswith("Returns") or var.strip() == "":
                        break

                    if ":" in var:
                        types.append(var.split(":")[-1].strip())
                    else:
                        types.append("str")

                    if doc_lines and doc_lines[0].startswith("    "):
                        help_doc.append(doc_lines.pop(0).strip())
                    else:
                        help_doc.append("")
            else:
                types = ["str"] * len(arg_spec.args)
                help_doc = [None] * len(arg_spec.args)

            return [
                ArgInfo(n, t, r, d, o)
                for n, t, r, d, o in zip(
                    arg_spec.args, types, required, defaults, help_doc
                )
            ]

        func._cli_command = command_name
        func._cli_service = service_name
        func._cli_arguments = parse_args

        return func

    if callable(name):
        # allow direct decoration without arguments
        return decorator(name)
    return decorator


def _find_services(module="sasctl"):
    """Recursively find all functions in all modules that have been decorated as CLI commands."""
    m = __import__(module, fromlist=[""])  # returns a module

    def find_recurse(module, services):
        for obj in dir(module):
            obj = getattr(module, obj)

            source_module = getattr(obj, "__module__", type(obj).__module__)

            # Module-level functions that are tagged as commands
            if hasattr(obj, "_cli_command") and hasattr(obj, "_cli_service"):
                services[obj._cli_service][obj._cli_command] = obj

            # Check methods on service classes
            elif source_module.startswith("sasctl._services"):
                for atr in dir(obj):
                    atr = getattr(obj, atr)
                    if hasattr(atr, "_cli_command") and hasattr(atr, "_cli_service"):
                        services[atr._cli_service][atr._cli_command] = atr

        # recurse into submodules
        submodules = pkgutil.iter_modules(getattr(module, "__path__", []))

        for submodule in submodules:

            # ModuleInfo returned py 3.6 has .name
            # Tuple of (module_loader, name, ispkg) returned by older versions
            submodule_name = getattr(submodule, "name", submodule[1])

            # TODO: Temporary until pzmm fully merged with sasctl
            if submodule_name == "pzmm":
                continue

            submodule = import_module("." + submodule_name, package=module.__name__)
            # if hasattr(submodule, 'name'):
            #     # ModuleInfo returned py 3.6
            #     submodule = import_module('.' + submodule.name, package=module.__name__)
            # else:
            #     # Tuple of (module_loader, name, ispkg) returned by older versions
            #     submodule = import_module('.' + submodule[1], package=module.__name__)
            services = find_recurse(submodule, services)

        return services

    services = find_recurse(m, defaultdict(dict))
    return services


def _get_func_description(func):
    description = getattr(func, "__doc__", "")

    lines = description.split("\n")

    if lines:
        return lines[0]


def _build_parser(services):
    from sasctl import __version__

    # TODO: Set command docstring

    # Create standard, top-level arguments
    parser = argparse.ArgumentParser(
        prog="sasctl", description="sasctl interacts with a SAS Viya environment."
    )
    parser.add_argument(
        "-k", "--insecure", action="store_true", help="skip SSL verification"
    )
    parser.add_argument(
        "-f", "--format", choices=["json"], default="json", help="output format"
    )
    parser.add_argument("-v", "--verbose", action="count")
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )

    subparsers = parser.add_subparsers(title="service", dest="service")
    subparsers.required = True

    for service, commands in services.items():
        service_parser = subparsers.add_parser(service)
        service_subparser = service_parser.add_subparsers(
            title="command", dest="command"
        )
        service_subparser.required = True

        # Add the command and arguments for each command
        for command in commands:
            func = services[service][command]

            cmd_parser = service_subparser.add_parser(
                command, help=_get_func_description(func)
            )

            for arg in func._cli_arguments():
                if arg.name in ("self", "cls"):
                    continue

                if arg.required:
                    cmd_parser.add_argument(arg.name, help=arg.doc)
                else:
                    cmd_parser.add_argument(
                        "--" + arg.name,
                        required=arg.required,
                        default=arg.default,
                        help=arg.doc,
                    )

    return parser


def main(args=None):
    """Main entry point when executed as a command line utility."""
    from sasctl import Session, current_session

    # Find all services and associated commands
    services = _find_services()

    parser = _build_parser(services)
    args = parser.parse_args(args)

    if args.verbose is None or args.verbose == 0:
        lvl = logging.WARNING
    elif args.verbose == 1:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG

    handler = logging.StreamHandler()
    handler.setLevel(lvl)
    logging.getLogger("sasctl.core").addHandler(handler)
    logging.getLogger("sasctl.core").setLevel(lvl)

    warnings.simplefilter("ignore")

    func = services[args.service][args.command]
    kwargs = vars(args).copy()

    # Remove args that shouldn't be passed to the underlying command
    for k in ["command", "service", "insecure", "verbose", "format"]:
        kwargs.pop(k, None)

    username = os.environ.get("SASCTL_USER_NAME")
    password = os.environ.get("SASCTL_PASSWORD")
    server = os.environ.get("SASCTL_SERVER_NAME")

    if server is None:
        parser.error(
            "Hostname must be specified in the 'SASCTL_SERVER_NAME' environment variable."
        )

    verify_ssl = not args.insecure

    try:
        #  current_session() should never be set when executing from the
        #  command line but it allows us to provide a pre-created session
        #  during testing
        with current_session() or Session(
            server, username, password, verify_ssl=verify_ssl
        ):
            result = func(**kwargs)
            if isinstance(result, list):
                pprint([str(x) for x in result])
            elif isinstance(result, dict) and args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                pprint(result)
    except RuntimeError as e:
        parser.error(e)
