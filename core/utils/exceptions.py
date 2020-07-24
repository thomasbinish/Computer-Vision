class UploadError(Exception):
    args = "Couldn't UPLOAD specified file"


class UpdateError(Exception):
    args = "Couldn't UPDATE specified file"


class DownloadError(Exception):
    args = "Couldn't DOWNLOAD specified file"


class FileNotFound(Exception):
    args = "Couldn't find file"
