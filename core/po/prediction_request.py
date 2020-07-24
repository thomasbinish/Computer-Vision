class PredictionRequest:
    def __init__(self, params):
        self.service = params["service"]
        self.name = params["name"]
        self.zip_file_url = params["zipFolderUrl"]
        self.model_url = params["modelUrl"]
        self.label = "vision"
