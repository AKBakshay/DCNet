import torch
import torchvision.datasets
from src.process.image_processing import clear_image, feature_maps, show_image
from torch.serialization import save
from torch.utils import data
from torchvision.transforms.transforms import ToTensor
from tqdm.std import tqdm


class Predictor:
    def __init__(self, model, transform, dataset, atm_light, cuda):
        self.model = model
        self.transform = transform
        self.dataset = dataset
        self.atm_light = atm_light
        self.cuda = cuda

    def predict(self):
        dataset = torchvision.datasets.ImageFolder(self.dataset, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        for input_data, _ in tqdm(dataloader, desc="Prediction"):
            input_data, hazy_data = feature_maps(
                input=input_data,
                cuda=self.cuda,
            )

            output_data = self.model(input_data)
            output_data = clear_image(output_data, hazy_data, self.atm_light)
            show_image(output_data[0])
