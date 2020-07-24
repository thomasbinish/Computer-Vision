class JobNotFoundError(Exception):
    args = "Job Not Found"

class FolderFormatError(Exception):
    args = "Extracted Zip folder should contain single folder"