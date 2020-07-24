class ImagePredictionRequest:
    def __init__(self, params):
        self.service = params["service"]
        self.zip_file_url = params["zipFolderUrl"]
        self.model_url = params["modelUrl"]
