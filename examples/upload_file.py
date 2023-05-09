import getpass
import os

from sasctl import Session, core
from sasctl.services import files, folders

server_name = "myServer.sas.com"
user = "sasdemo"
password = getpass.getpass()
os_path = "/test/Downloads/pdfs"
sas_path = "/Public/PDFs"


def build_storage_lists(os_path, sas_path):
    file_list = []
    dir_list = []

    for root, dirs, src_files in os.walk(os_path):
        for src_file in src_files:
            if src_file.endswith(".pdf"):
                if root not in dir_list:
                    dir_info = root.replace(os_path, sas_path)
                    dir_list.append(dir_info)
                file_info = {}
                file_info["source_file"] = os.path.join(root, src_file)
                file_info["target_folder"] = root.replace(os_path, sas_path)
                file_list.append(file_info)
    return dir_list, file_list


with Session(server_name, user , password ):
    dir_list, file_list = build_storage_lists(os_path, sas_path)
    for folder in dir_list:
        folders.create_folder_recursive(folder)
    for file in file_list:
        files.create_file(file["source_file"], file["target_folder"])
