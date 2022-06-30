#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Score Execution API is used to produce a score by executing the 
mapped code generated by score objects using the score definition."""

from .service import Service
from ..core import RestObj

class ScoreExecution(Service):
    """Implements the Score Execution REST API.
    
    The Score Execution API is used to validate a score object. First, a 
    Score Definition is created using the score object, the input data, and 
    the mappings. After the score execution has successfully completed, the 
    score output table can be validated for the expected values. Currently, 
    the validation is done manually. Further analysis can be done on the 
    score output table by generating an analysis output table.

    The Score Execution API can also generate the score output table by 
    providing mapped code directly instead of generating mapped code from 
    the Score Definition.
    
    See Also
    --------
    `REST Documentation <https://developer.sas.com/apis/rest/DecisionManagement
    /#score-execution>`_
    
    """
    
    _SERVICE_ROOT = "/scoreExecution"
    
    @classmethod
    def get_score_executions(cls):
        '''Returns a list of the score executions based on the specified pagination,
        filtering, and sorting options. The items in the list depend on the `Accept-Item`
        header.
        
        Returns
        -------
        RestObj
        '''
        headers = {"Accept": "application/vnd.sas.collection+json",
                   "Accept-Item": "application/vnd.sas.summary+json"}
        return cls.get("/executions",
                       headers=headers)
        
    @classmethod
    def create_score_execution(cls):
        '''Creates a new score execution based on the representation in the body.
        
        Parameters
        ----------
        
        Returns
        -------
        RestObj
        '''
        headers = {"Content-Type": "application/json",
                   "Accept": "application/vnd.sas.score.execution+json"}
        body = {}
        return cls.post("/executions",
                        data=body,
                        headers=headers)
        
    @classmethod
    def get_score_execution(cls, execution):
        '''Returns the representation of the specified score execution.

        Parameters
        ----------
        execution : _type_
            _description_
        
        Returns
        -------
        RestObj
        '''
        # Check if execution is an API response or UUID
        if type(execution)==RestObj:
            executionId = execution.id
        else:
            executionId = execution
        headers = {"Accept": "application/vnd.sas.score.execution+json"}
        return cls.get("/executions/{}".format(executionId),
                       headers=headers)
    
    @classmethod
    def delete_score_execution(cls, execution):
        '''Deletes the specified score execution.

        Parameters
        ----------
        execution : _type_
            _description_
            
        Returns
        -------
        RestObj
        '''
        # Check if execution is an API response or UUID
        if type(execution)==RestObj:
            executionId = execution.id
        else:
            executionId = execution
        return cls.delete("/executions/{}".format(executionId))