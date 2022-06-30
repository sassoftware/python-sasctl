#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Score Definitions API is used for creating and maintaining score definitions."""

from .service import Service
from ..core import RestObj

class ScoreDefinitions(Service):
    """Implements the Score Definiitions REST API.
    
    Score definitions are used by the Score Execution API to generate mapped code, which is then used to generate the score for the input data.

    A score definition contains the following details:

    1. Input data that needs to be scored.
    2. Details about the score object whose logic is used to produce a score.
    3. Mappings between the input data columns and the variables of the mapped code of the score object.
    
    See Also
    --------
    `REST Documentation <https://developer.sas.com/apis/rest/DecisionManagement
    /#score-definitions>`_
    """
    
    _SERVICE_ROOT = "/scoreDefinitions"
    
    @classmethod
    def get_score_definitions(cls):
        '''Returns a list of the score definitions based on the specified pagination,
        filtering, and sorting options. The items in the list depend on the `Accept-Item`
        header.
        
        Returns
        -------
        RestObj
        '''
        headers = {"Accept": "application/vnd.sas.collection+json",
                   "Accept-Item": "application/vnd.sas.summary+json"}
        return cls.get("/definitions",
                       headers=headers)
        
    @classmethod
    def create_score_definition(cls):
        '''Creates a new score definition based on the representation of the request body.
        
        Parameters
        ----------
        
        Returns
        -------
        RestObj
        '''
        headers = {"Content-Type": "application/vnd.sas.score.definition+json",
                   "Accept": "application/vnd.sas.score.definition+json"}
        body = {}
        return cls.post("/definitions",
                        data=body,
                        headers=headers)
        
    @classmethod
    def get_score_definition(cls, definition):
        '''Returns the representation of the specified score definition.

        Parameters
        ----------
        definition : _type_
            _description_

        Returns
        -------
        RestObj
        '''
        # Check if definition is an API response or UUID
        if type(definition)==RestObj:
            definitionId = definition.id
        else:
            definitionId = definition
        headers = {"Accept": "application/vnd.sas.score.definition+json"}
        return cls.get("/definitions/{}".format(definitionId),
                       headers=headers)
        
    @classmethod
    def delete_score_definition(cls, definition):
        '''Deletes the specified score definition.

        Parameters
        ----------
        definition : _type_
            _description_
        
        Returns
        -------
        RestObj
        '''
        # Check if definition is an API response or UUID
        if type(definition)==RestObj:
            definitionId = definition.id
        else:
            definitionId = definition
        return cls.delete("/definitions/{}".format(definitionId))