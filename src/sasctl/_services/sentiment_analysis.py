#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .service import Service
from ..core import current_session, uri_as_str


class SentimentAnalysis(Service):
    """The Sentiment Analysis API is used to perform sentiment analysis on
    natural language text documents according to a prebuilt or user-defined
    model.
    """

    _SERVICE_ROOT = "/sentiment"

    @classmethod
    def analyze_sentiment(
        cls,
        documents,
        caslib=None,
        id_column=None,
        text_column=None,
        description=None,
        model=None,
        output_postfix=None,
        language="en",
    ):
        """Performs sentiment analysis on the input data.

        Creates a setiment analysis task that executes asynchronously.  There
        are two different interactions for sentiment analysis: analyzing
        documents in CAS tables and analyzing documents that are uploaded directly.

        Parameters
        ----------
        documents : str or dict or list_like:
            Documents to analyze.  May be either the URI to a CAS table where the
            documents are currently stored, or an iterable of strings containing
            the documents' text.
        caslib : str or dict, optional
            URI of a caslib in which the documents will be stored.  Required if
            `documents` is a list of strings.
        id_column : str, optional
            The column in `documents` that contains a unique id for each
            document.  Required if `documents` is a CAS table URI.
        text_column : str, optional
            The column in `documents` that contains the document text to
            analyze.  Required if `documents` is a CAS table URI.
        description : str, optional
            Description to add to the sentiment analysis job.
        model
        output_postfix : str, optional
            Text to be added to the end of all output table names.
        language : str, optional
            Two letter
            `ISO 639-1 <https://en.wikipedia.org/wiki/ISO_639>`_
            code indicating the source language.  Defaults to 'en'.

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
                "The Sentiment Analysis service was removed from Viya 4."
            )

        if documents is None:
            raise TypeError("`documents` cannot be None.")

        if isinstance(documents, (dict, str)):
            data = {
                "inputUri": uri_as_str(documents),
                "documentIdVariable": id_column,
                "textVariable": text_column,
                "version": 1,
            }
        else:
            data = {
                "caslibUri": uri_as_str(caslib),
                "documents": documents,
                "version": 1,
            }

        data.update(
            {
                "description": description,
                "language": language,
                "modelUri": uri_as_str(model),
                "outputTableNamePostfix": output_postfix,
            }
        )

        # Optional fields are not ignored if None. Explicitly remove before sending
        for k in list(data.keys()):
            if data[k] is None:
                del data[k]

        url = "/jobs"

        # Update URL if passing in raw documents.
        if "documents" in data:
            url += "#data"
            headers = {
                "Content-Type": "application/vnd.sas.text.sentiment.job.request.documents+json",
                "Accept": "application/vnd.sas.text.sentiment.job+json",
            }
        else:
            headers = {
                "Content-Type": "application/vnd.sas.text.sentiment.job.request+json",
                "Accept": "application/vnd.sas.text.sentiment.job+json",
            }

        return cls.post(url, json=data, headers=headers)
