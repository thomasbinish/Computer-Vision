import requests
import tempfile
from properties import REGISTRY
if REGISTRY:
    from core.constants import STORAGE_URL
else:
    STORAGE_URL='http://127.0.0.1'
from core.utils.exceptions import *


def file_upload(file, user_id, label):
    url = STORAGE_URL + "/dltk-pdatestorage/s3/file"
    files = {'file': open(file, 'rb')}
    headers = {"user-id": user_id, "label": label}
    response = requests.post(url=url, files=files, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(response.json())
        raise UploadError()


def file_update(file, file_url):
        url = STORAGE_URL + "/dltk-storage/s3/file"
        files = {'file': open(file, 'rb')}
        data = {"key": file_url}
        response = requests.put(url=url, files=files, data=data)
        if response.status_code == 200:
            return response.text
        else:
            print(response.text)
            raise UpdateError()


def file_download(file_url, user_id):
    url = STORAGE_URL + "/dltk-storage/s3/file/download"
    data = {"url": file_url}
    print(file_url)
    headers= {"user-id":user_id}
    response = requests.get(url=url, params=data, headers = headers)
    if response.status_code == 200:
        temp = tempfile.NamedTemporaryFile(delete=False)
        f = open(temp.name, 'wb')
        f.write(response.content)
        f.close()
        temp.close()
        return temp.name
    else:
        print(response.text)
        raise DownloadError()
