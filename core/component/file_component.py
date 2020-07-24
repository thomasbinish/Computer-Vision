import shutil
from zipfile import ZipFile
import os

from core.exceptions import FolderFormatError
from core.models import ModelInfo
from core.utils.storage import file_download


def extract_data(zip_folder_path, worker):
    with ZipFile(zip_folder_path, 'r') as zipObj:
        zipObj.extractall()
    os.remove(zip_folder_path)
    folder = zipObj.namelist()[0].split("/")[0]
    if folder == "":
        raise FolderFormatError()
    print(folder)
    if "__MACOSX" in os.listdir(folder):
        shutil.rmtree(folder+"/__MACOSX")
    os.renames(folder.replace("/", ""), folder.replace("/", "") + "_" + str(worker))
    return folder.replace("/", "") + "_" + str(worker)


def extract_model(model_url, user_id):
    model_file = ModelInfo.objects.filter(model_url_s3=model_url, user_id=user_id).values()
    print(model_file)
    if len(model_file) == 0:
        info = ModelInfo()
        info.user_id = user_id
        info.model_url_s3 = model_url
        info.save()

        model = file_download(model_url, user_id)
        with ZipFile(model, 'r') as zipObj:
            zipObj.extractall('models/')
        os.remove(model)
        folder = "models/"+ zipObj.namelist()[0].split("/")[0]
        if folder == "":
            raise FolderFormatError()
        print(folder)
        if "__MACOSX" in os.listdir(folder):
            shutil.rmtree(folder+"/__MACOSX")

        os.renames(folder, folder + "_" + str("1"))
        dir_name = folder+"_"+str(1)
        info.model_path = dir_name
        info.save()
    else:
        model = model_file[0]
        dir_name = model["model_path"]
    return dir_name