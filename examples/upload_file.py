import getpass
import os

from sasctl import Session
from sasctl.services import files, folders

server_name = "myServer.sas.com"
user = "sasdemo"
password = getpass.getpass()
os_path = "/test/Downloads/pdfs"
sas_path = "/Public/PDFs"


def build_storage_lists(source_path, target_path):
    """
    Function to create a list of files and directories
    from a source folder at OS level
    and mapped to target folder in SAS Content Server.

    Parameters
    ----------
    source_path : str
        Path to the folder which should be imported.
    target_path : str
        Path to the folder which will contain the files and folders in SAS Content Server.

    Returns
    -------
    f_list
        A list of file information which can be used to map local and SAS files.
    d_list
        A list of directory information which can be user to map local and SAS folder structure.
    """
    f_list = []
    d_list = []

    for root, _, src_files in os.walk(source_path):
        for src_file in src_files:
            if src_file.endswith(".pdf"):
                if root not in dir_list:
                    dir_info = root.replace(source_path, target_path)
                    d_list.append(dir_info)
                file_info = {}
                file_info["source_file"] = os.path.join(root, src_file)
                file_info["target_folder"] = root.replace(
                    source_path, target_path)
                f_list.append(file_info)
    return d_list, f_list


with Session(server_name, user, password):
    dir_list, file_list = build_storage_lists(os_path, sas_path)
    for folder in dir_list:
        folders.create_folder_recursive(folder)
    for file in file_list:
        files.create_file(
            file["source_file"],
            file["target_folder"],
            os.path.basename(file["source_file"])
