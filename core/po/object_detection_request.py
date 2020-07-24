class ObjDetectionRequest:
    def __init__(self, params):
        self.service = "obj_detection"
        self.name = params["name"]
        self.zip_file_url = params["zipFolderUrl"]
        self.images = params["imagesColumn"]
        self.labels = params["labelsColumn"]
        self.label = "vision"
        self.steps = params["steps"]