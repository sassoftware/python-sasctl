#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .service import Service
from ..core import current_session, uri_as_str


class TextCategorization(Service):
    """Categorizes natural language text documents according to a prebuilt or
    user-defined model.
    """

    _SERVICE_ROOT = "/categorization"

    @classmethod
    def categorize(
        cls,
        documents,
        model,
        caslib=None,
        id_column=None,
        text_column=None,
        description=None,
        output_postfix=None,
    ):
        """

        Parameters
        ----------
        documents : str or dict or list_like:
            Documents to parse.  May be either the URI to a CAS table where the
            documents are currently stored, or an iterable of strings
            containing the documents' text.
        model : str or dict
            URI of a CAS table that contains one or more category model
            binaries.
        caslib : str or dict, optional
            URI of a caslib in which the documents will be stored.  Required if
            `documents` is a list of strings.
        id_column : str, optional
            The column in `documents` that contains a unique id for each
            document.  Required if `documents` is a CAS table URI.
        text_column : str, optional
            The column in `documents` that contains the document text to
            categorize.
            Required if `documents` is a CAS table URI.
        description : str, optional
            Description to add to the text categorization job.
        output_postfix : str, optional
            Text to be added to the end of all output table names.

        Returns
        -------
        RestObj
            The submitted job

        See Also
        --------
        :meth:`cas_management.get_caslib <.CASManagement.get_caslib>`
        :meth:`cas_management.get_table <.CASManagement.get_table>`

        """
        if current_session().version_info() >= 4:
            raise RuntimeError(
                "The Text Categorization service was removed from Viya 4."
            )

        if documents is None:
            raise TypeError("`documents` cannot be None.")

        url = "/jobs"

        if isinstance(documents, (dict, str)):
            # Input is caslib
            data = {
                "inputUri": uri_as_str(documents),
                "documentIdVariable": id_column,
                "textVariable": text_column,
                "version": 1,
            }
            headers = {
                "Content-Type": "application/vnd.sas.text.categorization.job.request+json",
                "Accept": "application/vnd.sas.text.categorization.job+json",
            }
        else:
            # Input is inline documents
            data = {
                "caslibUri": uri_as_str(caslib),
                "documents": documents,
                "version": 1,
            }
            url += "#data"
            headers = {
                "Content-Type": "application/vnd.sas.text.categorization.job.request.documents+json",
                "Accept": "application/vnd.sas.text.categorization.job+json",
            }

        data.update(
            {
                "description": description,
                "modelUri": uri_as_str(model),
                "outputTableNamePostfix": output_postfix,
            }
        )

        # Optional fields are not ignored if None. Explicitly remove before sending
        for k in list(data.keys()):
            if data[k] is None:
                del data[k]

        return cls.post(url, json=data, headers=headers)
