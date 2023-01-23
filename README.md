<div align="center">  

  <h1>sasctl</h1>

  <p>A user-friendly Python interface for SAS Viya.</p>
  
  <a href="https://www.sas.com/en_us/software/viya.html">
    <img src="https://img.shields.io/badge/SAS%20Viya-3.5+-blue.svg?&colorA=0b5788&logoWidth=30&logo=data:image/png;base64,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"
        alt="SAS Viya Version"/>
  </a>
        
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.6%2B-blue.svg" alt="Python Version">
  </a>

  <a href="https://github.com/sassoftware/python-sasctl/actions/workflows/build-test-deploy.yml">
    <img src="https://github.com/sassoftware/python-sasctl/actions/workflows/build-test-deploy.yml/badge.svg" />
  </a>
    
  <a href="https://codecov.io/gh/sassoftware/python-sasctl">
    <img src="https://codecov.io/gh/sassoftware/python-sasctl/branch/master/graph/badge.svg?token=mDcvtz2Als"/>
  </a>

    
</div>



###### Full documentation:  https://sassoftware.github.io/python-sasctl

# Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [Examples](#examples)
6. [Contributing](#contributing)
7. [License](#license)
8. [Additional Resources](#additional-resources)


## Overview

The sasctl package enables easy communication between the SAS Viya 
platform and a Python runtime. It can be used as a module or as a command line interface.
```
sasctl.services.folders.list_folders()
```

```
sasctl folders list
```


### Prerequisites

sasctl requires the following Python packages be installed.
If not already present, these packages will be downloaded and installed automatically.  
- pandas
- requests
- pyyaml
- packaging

The following additional packages are recommended for full functionality:
- swat
- kerberos
- GitPython
- numpy
- scikit-learn


### Installation

Installing the latest version is as easy as:
```
pip install sasctl
```

Functionality that depends on additional packages can be installed using the following:
 - `pip install sasctl[swat]`
 - `pip install sasctl[kerberos]`
 - `pip install sasctl[all]`
  
If you want the latest functionality and can't wait on an official release, you can also
install the latest source code:

```pip install git+https://github.com/sassoftware/python-sasctl```


## Getting Started

Once the sasctl package has been installed and you have a SAS Viya server to connect to, 
the first step is to establish a session:
```
>>> from sasctl import Session

>>> with Session(host, username, password):
...     pass  # do something
```
```
sasctl --help 
```


Once a session has been created, all commands target that environment. 
The easiest way to use sasctl is often to use a pre-defined task, 
which can handle all necessary communication with the SAS Viya server:
```
>>> from sasctl import Session, register_model
>>> from sklearn import linear_model as lm

>>> with Session('example.com', authinfo=<authinfo file>):
...    model = lm.LogisticRegression()
...    register_model(model, 'Sklearn Model', 'My Project')
```


A slightly more low-level way to interact with the environment is to use 
the service methods directly:
```
>>> from sasctl import Session
>>> from sasctl.services import folders

>>> with Session(host, username, password):
...    for f in folders.list_folders():
...        print(f)

Public
Projects
ESP Projects
Risk Environments

...  # truncated for clarity

My Folder
My History
My Favorites
SAS Environment Manager
```


The most basic way to interact with the server is simply to call REST 
functions directly, though in general, this is not recommended.
```
>>> from pprint import pprint
>>> from sasctl import Session, get

>>> with Session(host, username, password):
...    folders = get('/folders')
...    pprint(folders)
    
{'links': [{'href': '/folders/folders',
            'method': 'GET',
            'rel': 'folders',
            'type': 'application/vnd.sas.collection',
            'uri': '/folders/folders'},
           {'href': '/folders/folders',
            'method': 'POST',
            'rel': 'createFolder',

...  # truncated for clarity

            'rel': 'createSubfolder',
            'type': 'application/vnd.sas.content.folder',
            'uri': '/folders/folders?parentFolderUri=/folders/folders/{parentId}'}],
 'version': 1}
```




### Examples

A few simple examples of common scenarios are listed below.  For more 
complete examples see the [examples](examples) folder.

Show models currently in Model Manager:
```
>>> from sasctl import Session
>>> from sasctl.services import model_repository

>>> with Session(host, username, password):
...    models = model_repository.list_models()
```

Register a pure Python model in Model Manager:
```
>>> from sasctl import Session, register_model
>>> from sklearn import linear_model as lm

>>> with Session(host, authinfo=<authinfo file>):
...    model = lm.LogisticRegression()
...    register_model(model, 'Sklearn Model', 'My Project')
```

Register a CAS model in Model Manager:
```
>>> import swat
>>> from sasctl import Session
>>> from sasctl.tasks import register_model

>>> s = swat.CAS(host, authinfo=<authinfo file>)
>>> astore = s.CASTable('some_astore')

>>> with Session(s):
...    register_model(astore, 'SAS Model', 'My Project')
```

## Contributing

We welcome contributions! 

Please read [CONTRIBUTING.md](CONTRIBUTING.md) 
for details on how to submit contributions to this project.

## License

See the [LICENSE](LICENSE) file for details.

## Additional Resources

* [SAS Viya REST Documentation](https://developer.sas.com/apis/rest/)
* [SAS Developer Community](https://communities.sas.com/t5/Developers/bd-p/developers)

