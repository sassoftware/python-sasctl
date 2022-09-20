#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enables publishing objects such as models to various destinations."""

import re

from .model_repository import ModelRepository
from .service import Service


class ModelPublish(Service):
    """Enables publishing objects such as models to various destinations.

    The service provides methods for managing publish destinations like CAS,
    Hadoop, Teradata, or SAS Micro Analytic Service
    """

    _SERVICE_ROOT = "/modelPublish"
    _model_repository = ModelRepository()

    @staticmethod
    def _publish_name(name):
        """Create a valid module name from an input string.

        Model Publishing only permits names that adhere to the following
        restrictions:
            - Name must start with a letter or an underscore.
            - Name may not contain any spaces or special characters other than
              underscore.

        Parameters
        ----------
        name : str

        Returns
        -------
        str

        """
        # Remove all non-word characters
        name = re.sub(r"[^\w]", "", str(name))

        # Add a leading underscore if name starts with a digit
        if re.match(r"\d", name):
            name = "_" + name

        return name

    @classmethod
    def list_models(cls):
        return cls.get("/models").get("items", [])

    list_destinations, get_destination, update_destination, _ = Service._crud_funcs(
        "/destinations", "destination"
    )

    # @sasctl_command('delete')
    @classmethod
    def delete_destination(cls, item):
        """Delete a destination instance.

        Parameters
        ----------
        item : str or dict
            Name or dictionary representation of publishing destination.

        Returns
        -------
        None

        """

        # NOTE:  Explicitly defined since service requires destination NAME not
        # ID which is standard.
        item_name = str(item)

        # Try to find the item if the id can't be found
        if not (isinstance(item, dict) and "id" in item):
            item = cls.get_destination(item)
            if item is None:
                cls.log.info("Object '%s' not found.  Skipping delete.", item_name)
                return

        if isinstance(item, dict) and "name" in item:
            item = item["name"]

        return cls.delete("/destinations/{name}".format(name=item))

    @classmethod
    def publish_model(cls, model, destination, name=None, code=None, notes=None):
        """Publish a model to an existing publishing destination.

        Parameters
        ----------
        model : str or dict
            The name or id of the model, or a dictionary representation of
            the model.
        destination : str or dict
            The name or id of the publishing destination, or a dictionary
            representation of the destination
        name : str, optional
            Name of the published model.  Defaults to the model name.
        code : str, optional
            The code to be published.
        notes

        Returns
        -------

        """
        code_types = {"ds2package": "ds2", "datastep": "datastep", "": ""}

        model = cls._model_repository.get_model(model)
        model_uri = cls._model_repository.get_model_link(model, "cls")

        # Get score code from registry if no code specified
        if code is None:
            code_link = cls._model_repository.get_model_link(model, "scoreCode", True)
            if code_link:
                code = cls.get(code_link["href"])

        request = dict(
            name=name or model.get("name"),
            notes=notes,
            destinationName=destination,
        )

        model_contents = {
            "modelName": model.get("name"),
            "modelId": model.get("id"),
            "sourceUri": model_uri.get("href"),
            "publishLevel": "model",  # ?? What are the options?
            "codeType": code_types[model.get("scoreCodeType", "").lower()],
            "codeUri": "",  # ??  Not needed if code is specified?
            "code": code,
        }

        request["model_contents"] = [model_contents]
        return cls.post(
            "/models",
            json=request,
            headers={
                "Content-Type": "application/vnd.sas.models.publishing.request+json"
            },
        )

    @classmethod
    def create_cas_destination(
        cls, name, library, table, server=None, description=None
    ):
        """Define a new CAS publishing destination.

        Parameters
        ----------
        name : str
            Name of the publishing destination.
        library : str
            The CAS library in which `table` will be stored.
        table : str
            Name of the CAS table in which models will be stored.
        server : str, optional
            Name of the CAS server.  Defaults to 'cas-shared-default'.
        description : str, optional
            Description of the publishing destination.

        Returns
        -------
        RestObj

        """
        server = server or "cas-shared-default"

        return cls.create_destination(
            name,
            cas_server=server,
            cas_library=library,
            cas_table=table,
            type_="cas",
            description=description,
        )

    @classmethod
    def create_mas_destination(cls, name, uri, description=None):
        """Define a new Micro Analytic Server (MAS) publishing destination.

        Parameters
        ----------
        name : str
            Name of the publishing destination.
        uri : str
            The base URI that contains the host, the protocol, and optionally
            the port, which addresses the remote SAS Micro Analytic Service to
            use.  Example: http://spam.com
        description : str, optional
            Description of the publishing destination.

        Returns
        -------
        RestObj

        """
        return cls.create_destination(
            name, mas_uri=uri, type_="mas", description=description
        )

    @classmethod
    def create_destination(
        cls,
        name,
        type_,
        cas_server=None,
        cas_library=None,
        cas_table=None,
        description=None,
        mas_uri=None,
        hdfs_dir=None,
        conf_dir=None,
        user=None,
        database_library=None,
    ):
        """Define a new publishing destination.

        Parameters
        ----------
        name : str
            Name of the publishing destination.
        type_ : {'cas', 'mas', 'hadoop', 'teradata'}
            Type of publishing definition being created
        cas_server : str, optional
            Name of the CAS server.  Defaults to 'cas-shared-default'.
            Required if `type_` is 'cas', otherwise ignored.
        cas_library : str, optional
            The CAS library in which `cas_table` will be stored.  Required if
            `type_` is 'cas', otherwise ignored.
        cas_table : str, optional
            Name of the CAS table in which models will be stored.  Required
            if `type_` is 'cas', otherwise ignored.
        description : str, optional
            Description of the publishing destination.
        mas_uri : str, optional
            Required if `type_` is 'mas', otherwise ignored.
        hdfs_dir : str, optional
            Required if `type_` is 'hadoop', otherwise ignored.
        conf_dir : str, optional
            Required if `type_` is 'hadoop', otherwise ignored.
        user : str, optional
            Required if `type_` is 'hadoop', otherwise ignored.
        database_library : str, optional
            Required if `type_` is 'teradata', otherwise ignored.

        Returns
        -------
        RestObj

        """
        type_ = str(type_).lower()
        if type_ not in ("cas", "microanalyticservice", "mas", "teradata", "hadoop"):
            raise ValueError("Unrecognized destination type '%s' specified." % type_)

        # As of Viya 3.4 capitalization matters.
        if type_ in ("microanalyticservice", "mas"):
            type_ = "microAnalyticService"

        request = {
            "name": str(name),
            "destinationType": type_,
            "casServerName": cas_server,
            "casLibrary": cas_library,
            "description": description,
            "destinationTable": cas_table,
            "databaseCasLibrary": database_library,
            "user": user,
            "hdfsDirectory": hdfs_dir,
            "configurationDirectory": conf_dir,
            "masUri": mas_uri,
        }

        drop_list = {
            "cas": (
                "databaseCasLibrary",
                "user",
                "hdfsDirectory",
                "masUri",
                "configurationDirectory",
            ),
            "microAnalyticService": (
                "casServerName",
                "casLibrary",
                "destinationTable",
                "user",
                "databaseCasLibrary",
                "hdfsDirectory",
                "configurationDirectory",
            ),
            "hadoop": (
                "casServerName",
                "casLibrary",
                "destinationTable",
                "databaseCasLibrary",
                "masUri",
            ),
            "teradata": (
                "casServerName",
                "casLibrary",
                "destinationTable",
                "user",
                "hdfsDirectory",
                "masUri",
                "configurationDirectory",
            ),
        }

        for k in drop_list[request["destinationType"]]:
            request.pop(k, None)

        return cls.post(
            "/destinations",
            json=request,
            headers={
                "Content-Type": "application/vnd.sas.models.publishing.destination+json"
            },
        )
