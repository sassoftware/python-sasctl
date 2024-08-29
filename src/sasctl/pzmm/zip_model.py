# Copyright (c) 2020, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
import zipfile
from pathlib import Path
from typing import Optional, Union


def _filter_files(file_dir: Union[str, Path], is_viya4: Optional[bool] = False) -> list:
    """
    Filters file list to only contain files used for model import. Models imported into
    SAS Viya 3.5 and SAS Viya 4 have a difference in total files imported, due to
    differences in Python handling.

    Parameters
    ----------
    file_dir : str or Path
        Location of \*.json, \*.pickle, \*.mojo, and \*Score.py files.
    is_viya4 : bool, optional
        Boolean to indicate difference in logic between SAS Viya 3.5 and SAS Viya 4. For
        Viya 3.5 models, ignore score code that is already in place in the file
        directory provided. Default value is False.

    Returns
    -------
    file_names : list
        Filtered list of file names to be uploaded in a SAS Viya model.
    """
    file_names = []
    file_names.extend(sorted(Path(file_dir).glob("*.json")))
    if is_viya4:
        file_names.extend(sorted(Path(file_dir).glob("score_*.py")))
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
    def zip_files(
        model_files: Union[dict, str, Path],
        model_prefix: str,
        is_viya4: Optional[bool] = False,
    ) -> BytesIO:
        """
        Combines all JSON files with the model pickle file and associated score code
        file into a single archive ZIP file.

        If the model_files argument is a string or Path object, then a zip file will
        be created at the directory location. Otherwise, the zip file is created in
        memory.

        Parameters
        ----------
        model_files : str, pathlib.Path, or dict
            Either the directory location of the model files (string or Path object), or
            a dictionary containing the contents of all the model files.
        model_prefix : str
            Variable name for the model to be displayed in SAS Open Model Manager
            (i.e. hmeqClassTree + [Score.py || .pickle]).
        is_viya4 : bool, optional
            Boolean to indicate difference in logic between SAS Viya 3.5 and SAS Viya 4.
            For Viya 3.5 models, ignore score code that is already in place in the file
            directory provided. Default value is False.
        """

        if isinstance(model_files, dict):
            buffer = BytesIO()

            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, False) as archive:
                for file_name, data in model_files.items():
                    if not isinstance(data, (str, bytes)):
                        data = str(data)
                    archive.writestr(file_name, data)

            # NOTE: bytes are added to the buffer when zip file is closed
            # so ensure closed before returning
            return buffer
        else:
            file_names = _filter_files(model_files, is_viya4)
            with zipfile.ZipFile(
                str(Path(model_files) / (model_prefix + ".zip")), mode="w"
            ) as zFile:
                for file in file_names:
                    zFile.write(str(file), arcname=file.name)

            with open(
                str(Path(model_files) / (model_prefix + ".zip")), "rb"
            ) as zip_file:
                return BytesIO(zip_file.read())
