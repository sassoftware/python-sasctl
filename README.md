# sasctl

## Overview

The sasctl package enables easy communication between the SAS Viya 
platform and a Python runtime. It can be used as a module or as a command line interface.
```
sasctl.folders.list_folders()
```

```
sasctl folders list
```


### Prerequisites

sasctl requires the following Python packages be installed.
If not already present, these packages will be downloaded and installed automatically.  
- requests
- six

The following additional packages are recommended for full functionality:
- swat
- kerberos / winkerberos


All required and recommended packages are listed in `requirements.txt` and can be installed easily with 
```
pip install -r requirements.txt
```

### Installation

```
pip install git+https://github.com/sassoftware/python-sasctl
```
  

## Getting Started

Read the full documentation here: [http://xeno.glpages.sas.com/python-sasctl](http://xeno.glpages.sas.com/python-sasctl)

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
...    register_model('Sklearn Model', model, 'My Project')
```


A slightly more low-level way to interact with the environment is to use 
the service methods directly:
```
>>> from pprint import pprint
>>> from sasctl import Session, folders

>>> with Session(host, username, password):
...    folders = folders.list_folders()
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
>>> from sasctl import Session, model_repository

>>> with Session(host, username, password):
...    models = model_repository.list_models()
```

Register a pure Python model in Model Manager:
```
>>> from sasctl import Session, register_model
>>> from sklearn import linear_model as lm

>>> with Session(host, authinfo=<authinfo file>):
...    model = lm.LogisticRegression()
...    register_model('Sklearn Model', model, 'My Project')
```

Register a CAS model in Model Manager:
```
>>> import swat
>>> from sasctl import Session
>>> from sasctl.tasks import register_model

>>> s = swat.CAS(host, authinfo=<authinfo file>)
>>> astore = s.CASTable('some_astore')

>>> with Session(s):
...    register_model('SAS Model', astore, 'My Project')
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

