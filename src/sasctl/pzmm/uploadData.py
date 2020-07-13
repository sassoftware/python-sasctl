# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# %%
from pathlib import Path

import requests
import getpass
import json

# %%

def getAccessToken(server, username=None, password=None):
    '''
    Retrieves an access token from the host server for further API requests. There
    are two options for retrieving an access token:
        1. Provide only the host server in the arguments. The user is
        prompted to input their user name and password. The password is not
        displayed in the prompt, nor kept in memory. Five attempts are
        allowed before the program exits.
        2. Provide the user name, password, and host server in the arguments.
        After the authentication attempt, the password will be removed from
        the memory.
        
    Parameters
    ---------------
    server : string
        Name of the host server with a SAS Open Model Manager or SAS Model Manager installation,
        which includes the protocol specification (i.e. http://).
    username : string
        The user name that is used for authentication with the server.
    password : string
        The password that is  used for authentication with the server.
    
    Returns
    ---------------
    authToken : string
        Access token from the JSON file in the API post request. It can also be used for
        additional API requests to the host server.
    '''
    
    authURI = '/SASLogon/oauth/token'
    headersAuth = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic c2FzLmVjOg=='
            }
    authToken = ''
    
    if (username is not None) and (password is not None):
        authBody = ('grant_type=password&username=' + username +
                    '&password=' + password)
        authReturn = requests.post(server + authURI,
                                   data=authBody,
                                   headers=headersAuth)
        if authReturn.status_code == 200:
            authToken = authReturn.json()['access_token']        
            notAuthenticated = False
        else:
            notAuthenticated = True
            print('The specified user name and password could not be' +
                  ' authenticated. Please enter a valid user name and' +
                  ' password.')
    else:
        notAuthenticated = True
    
    loginAttempts = 0
    while (notAuthenticated and loginAttempts < 5): 
        username = input('Enter user name:')
        password = getpass.getpass('Enter password for %s:' % username)
        authBody = ('grant_type=password&username=' + username +
                    '&password=' + password)
        authReturn = requests.post(server + authURI,
                                   data=authBody,
                                   headers=headersAuth)
        if authReturn.status_code == 200:
            authToken = authReturn.json()['access_token']
            notAuthenticated = False
        else:
            print('Please enter a valid user name and password.')
            print('The returned status code was %d.' % authReturn.status_code)
            loginAttempts += 1
            if loginAttempts == 5:
                print('No remaining attempts. Exiting program.')
                return False
            else:
                print('%d attempts left. \n' % (5-loginAttempts))

    password = ''
    
    return f'Bearer {authToken}'


class ModelImport():
    
    def __init__(self, host):
        '''
        Initializes the  ModelImport class with host location, user name, and password.
        
        Parameters
        ---------------
        host : string
            Name of the host server with a SAS Open Model Manager or SAS Model Manager installation.
        '''
        
        if host[:7] == 'http://':
            host = host[7:]
        self.host = host
        self.server = 'http://' + host
            
    def findProjectID(self, projectName, authToken):
        '''
        Given a project name, makes an API request to the Model Repository API to find the
        project ID. If project ID is not found, creates a new project and returns
        its project ID.
        
        Parameters
        ---------------
        projectName : string
            Project name for retrieving the project ID.
        authToken : string
            Access token that is used for API requests.
        
        Returns
        ---------------
        projectID : string
            The universally unique identifier string for a project.
        '''
        
        headers = {
                'Origin': self.server,
                'Authorization': authToken}
        requestUrl = f'{self.server}/modelRepository/projects?limit=100000'
        projectRequest = requests.get(requestUrl, headers=headers)
        
        projectID = [x['id'] for x in projectRequest.json()['items'] if x['name']==projectName]
        if not projectID:
            print(f'A project with the name "{projectName}" could not be found.')
            print(f'A new project with the name "{projectName}" is being created.')
            projectID = self.createNewProject(projectName, authToken)
            return projectID
        else:
            return projectID

    def createNewProject(self, projectName, authToken):
        '''
        Determines the Public repository folder ID, and then creates a new project 
        in the common model repository to store the models.
        
        Parameters
        ---------------
        projectName : string
            The project name used for retrieving the project ID.
        authToken : string
            The access token that is used for API requests.
        
        Returns
        ---------------
        projectID : string
            The universally unique identifier string for the project.
        '''
        
        repositoryHeaders = {'Authorization': authToken}
        requestUrl = (f'{self.server}/modelRepository/repositories' +
                      "?filter=eq(name,'Public')")
        repositoryList = requests.get(requestUrl, headers=repositoryHeaders)
        repositoryID = repositoryList.json()['items'][0]['id']
        repositoryFolderID = repositoryList.json()['items'][0]['folderId']
        
        headers = {'Content-Type': 'application/vnd.sas.models.project+json',
                   'Authorization': authToken}
        body = {'name': projectName,
                'repositoryId': repositoryID,
                'folderId': repositoryFolderID}
        url = f'{self.server}/modelRepository/projects'
        newProject = requests.post(url, data=json.dumps(body), headers=headers)
        
        return newProject.json()['id']
    
    def importModel(self, modelPrefix, projectID=None,
                    projectName=None, zPath=Path.cwd(),
                    username=None, password=None):
        '''
        Imports the zipped pickle file and corresponding Python and JSON files into
        the common model repository using the 'import model' API request.
        
        If the project ID is not known, provide the project name and an API
        request searches for the project ID. If a project does not already exist, and you
        do not provide either the project name or ID, the function  
        creates a new project via an API request.
        
        Parameters
        ---------------
        modelPrefix : string
            The variable for the model name that is used when naming the model files
            (i.e. hmeqClassTree + [Score.py || .pickle]).
        projectID : string, optional
            The universally unique identifier string for the project. The default value is None.
        projectName : string, optional
            The project name for retrieving the project ID. The default value is None.
        zPath : string, optional
            Location for the archive ZIP file. The default value is the current
            working directory.
        username : string
            The user name that is used for authentication with the server.
        password : string
            The password that is used for authentication with the server.
        '''
        authToken = getAccessToken(self.server, username, password)
        
        if not authToken:
            return

        if projectID is None and projectName is not None:
            projectID = self.findProjectID(projectName, authToken)
        elif projectID is None and projectName is None:
            projectName = input('Enter a new project name:')
            projectID = self.createNewProject(projectName, authToken)
        
        with open(zPath, 'rb') as zFile:
            
            body = {'name': modelPrefix,
                    'type': 'zip',
                    'projectId': projectID,
                    'versionOption': 'LATEST'}

            files = {'file': (f'{modelPrefix}.zip',
                              zFile,
                              'multipart/form-data')}

            headers = {'Origin': f'{self.server}',
                       'Authorization': authToken}

            url = f'{self.server}/modelRepository/models'
            modelRequest = requests.post(url,
                                         headers=headers,
                                         files=files,
                                         data=body)
        
        try:
            modelRequest.raise_for_status()
        except requests.exceptions.HTTPError:
            print('The model could not be imported: ' +
                  f'A model with the name "{modelPrefix}" already exists.')
            print('Enter a unique file name for the model ZIP file.')

# The following code is obsolete and was deprecated after the November 2019 release.
#    def uploadPickle(pLocalPath, pRemotePath,
#                     host, username, password=None, privateKey=None):
#        TODO: Remove password from memory as in self.getAccessToken()
#        '''
#        Uploads a local pickle file to a host server via sftp. Set the
#        permission of the pickle file on the server to 777 to allow the score
#        code to use the pickle file.
#        
#        Parameters
#        ---------------
#        pLocalPath : string
#            The local path for the pickle file.
#        pRemotePath : string
#            The remote path on the server for the pickle file's location.
#        host : string
#            The name of the host server to send the pickle file.
#        username : string
#            The server login credential user name.
#        password : string, optional
#            The password for SFTP connection attempt. The default value is None, in the case
#            where the user is using an RSA/DSA key pairing.
#        privateKey : string, optional
#            The private key location for the RSA/DSA key pairing logins. The default value is
#            None.
#        '''
#        
#        # convert windows path format to linux path format
#        if platform.system() == 'Windows':
#            pRemotePath = ('/' + 
#                           os.path.normpath(pRemotePath).replace('\\', '/'))
#        
#        with pysftp.Connection(host, username=username, password=password,
#                               private_key=privateKey) as sftp:
#            sftp.put(pLocalPath, remotepath=pRemotePath)
#            sftp.chmod(pRemotePath, mode=777)