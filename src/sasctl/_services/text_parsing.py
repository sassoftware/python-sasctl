#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .service import Service
from sasctl.core import current_session, uri_as_str


class TextParsing(Service):
    """The Text Parsing API parses natural language text documents.

    Parsing is a key operation in understanding your data. Parsing a document
    involves the following analyses:

     - Identifying terms used in the document
     - Recognizing parts of speech for each term
     - Identifying which terms are entities (person, country, and so on)
     - Resolving synonyms, misspellings, and so on

    The output tables that are generated during parsing can also be used in
    downstream analyses such as topic generation.

    """

    _SERVICE_ROOT = "/parsing"

    @classmethod
    def parse_documents(
        cls,
        documents,
        caslib=None,
        id_column=None,
        text_column=None,
        description=None,
        standard_entities=False,
        noun_groups=False,
        min_doc_count=10,
        concept_model=None,
        output_postfix=None,
        spell_check=False,
        override_list=None,
        stop_list=None,
        start_list=None,
        synonym_list=None,
        language="en",
    ):
        """Performs natural language parsing on the input data.

        Creates a text parsing job that executes asynchronously.  There are two
        different interactions for parsing: parsing documents in CAS tables and
        parsing documents that are uploaded directly.

        Parameters
        ----------
        documents : str or dict or list_like:
            Documents to parse.  May be either the URI to a CAS table where the
            documents are currently stored, or an iterable of strings containing
            the documents' text.
        caslib : str or dict, optional
            URI of a caslib in which the documents will be stored.  Required if
            `documents` is a list of strings.
        id_column : str, optional
            The column in `documents` that contains a unique id for each
            document.  Required if `documents` is a CAS table URI.
        text_column : str, optional
            The column in `documents` that contains the document text to parse.
            Required if `documents` is a CAS table URI.
        description : str, optional
            Description to add to the text parsing job.
        standard_entities : bool, optional
        noun_groups : bool, optional
        min_doc_count : int, optional
            Minimum number of documents in which a term must appear to be kept.
            Defaults to 10.
        output_postfix : str, optional
            Text to be added to the end of all output table names.
        spell_check : bool, optional
            Whether spell checking should be performed during parsing.
        concept_model : str or dict, optional
            URI of a table containing the concept LITI binaries to apply during
            parsing.
        override_list : str or dict, optional
            URI of a table containing overrides for the keep and drop terms.
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
            raise RuntimeError("The Text Parsing service was removed from Viya 4.")

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
                "includeStandardEntities": standard_entities,
                "includeNounGroups": noun_groups,
                "startListUri": uri_as_str(start_list),
                "stopListUri": uri_as_str(stop_list),
                "synonymListUri": uri_as_str(synonym_list),
                "minimumDocumentCount": min_doc_count,
                "conceptModelUri": uri_as_str(concept_model),
                "outputTableNamePostfix": output_postfix,
                "enableSpellChecking": spell_check,
                "overrideListUri": uri_as_str(override_list),
            }
        )

        # Optional fields are not ignored if None so explicitly remove before
        # sending.
        for k in list(data.keys()):
            if data[k] is None:
                del data[k]

        url = "/jobs"

        # Update URL if passing in raw documents.
        if "documents" in data:
            url += "#data"
            headers = {
                "Content-Type": "application/vnd.sas.text.parsing.job.request.documents+json",
                "Accept": "application/vnd.sas.text.parsing.job+json",
            }
        else:
            headers = {
                "Content-Type": "application/vnd.sas.text.parsing.job.request+json",
                "Accept": "application/vnd.sas.text.parsing.job+json",
            }

        return cls.post(url, json=data, headers=headers)
