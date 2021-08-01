import src.common.tools as tools
import torch
import torchvision.datasets
from src.process.image_processing import clear_image, feature_maps, save_image
from torch.serialization import save
from torch.utils import data
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm


class Predictor:
    def __init__(self, model, transform, dataset, atm_light, cuda, prediction_dir):
        self.model = model
        self.transform = transform
        self.dataset = dataset
        self.atm_light = atm_light
        self.cuda = cuda
        self.prediction_dir = prediction_dir

    def predict(self):
        dataset = torchvision.datasets.ImageFolder(self.dataset, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        # for i, (input_data, _) in tqdm(dataloader, desc="Prediction"):
        iter = 0
        for input_data, _ in tqdm(dataloader, desc="Prediction"):
            input_data, hazy_data = feature_maps(
                input=input_data,
                cuda=self.cuda,
            )
            iter += 1
            output_data = self.model(input_data)
            output_data = clear_image(output_data, hazy_data, self.atm_light)
            fname = tools.get_file_name(dataloader.dataset.samples[iter])
            save_image(output_data[0], self.prediction_dir, fname)
