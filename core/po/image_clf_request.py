class ImageClfRequest:
    def __init__(self, params):
        self.service = "classification"
        self.name = params["name"]
        self.zip_file_url = params["zipFolderUrl"]
        self.images = params["imagesColumn"]
        self.labels = params["labelsColumn"]
        self.train_percentage = params["trainPercentage"]
        self.label = "vision"
        self.epochs = params["epochs"]
        self.steps_per_epoch = params["stepsPerEpoch"]
