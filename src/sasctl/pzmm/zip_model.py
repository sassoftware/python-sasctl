# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import zipfile
import io


def _filter_files(file_dir, is_viya4=False):
    """
    Filters file list to only contain files used for model import. Models imported into SAS Viya 3.5 and SAS Viya 4 have
    a difference in total files imported, due to differences in Python handling.
    Parameters
    ----------
    file_dir : string
        Location of *.json, *.pickle, *.mojo, and *Score.py files.
    is_viya4 : boolean, optional
        Boolean to indicate difference in logic between SAS Viya 3.5 and SAS Viya 4. For Viya 3.5 models, ignore score
        code that is already in place in the file directory provided. Default value is False.

    Returns
    -------
    file_names : list
        Filtered list of file names to be uploaded in a SAS Viya model.
    """
    file_names = []
    file_names.extend(sorted(Path(file_dir).glob("*.json")))
    if is_viya4:
        file_names.extend(sorted(Path(file_dir).glob("*Score.py")))
    file_names.extend(sorted(Path(file_dir).glob("*.pickle")))
    # Include H2O.ai MOJO files
    file_names.extend(sorted(Path(file_dir).glob("*.mojo")))
    if file_names:
        return file_names
    else:
        raise FileNotFoundError(
            "No valid model files were found in the provided file directory."
        )


class ZipModel:
    @staticmethod
    def zip_files(file_dir, model_prefix, is_viya4=False):
        """
        Combines all JSON files with the model pickle file and associated score code file
        into a single archive ZIP file.

        Parameters
        ---------------
        file_dir : string
            Location of *.json, *.pickle, *.mojo, and *Score.py files.
        model_prefix : string
            Variable name for the model to be displayed in SAS Open Model Manager
            (i.e. hmeqClassTree + [Score.py || .pickle]).
        is_viya4 : boolean, optional
            Boolean to indicate difference in logic between SAS Viya 3.5 and SAS Viya 4.
            For Viya 3.5 models, ignore score code that is already in place in the file
            directory provided. Default value is False.

        Yields
        ---------------
        '*.zip'
            Archived ZIP file for importing into SAS Open Model Manager. In this form,
            the ZIP file can be imported into SAS Open Model Manager.
        """
        file_names = _filter_files(file_dir, is_viya4)
        with zipfile.ZipFile(
            str(Path(file_dir) / (model_prefix + ".zip")), mode="w"
        ) as zFile:
            for file in file_names:
                zFile.write(str(file), arcname=file.name)

        with open(str(Path(file_dir) / (model_prefix + ".zip")), "rb") as zip_file:
            return io.BytesIO(zip_file.read())
