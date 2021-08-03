import src.common.tools as tools
import torch
import torchvision.datasets
from src.process.image_processing import (
    add_haze,
    clear_image,
    feature_maps,
    generate_input,
    save_image,
)
from torch.serialization import save
from torch.utils import data
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm


class Predictor:
    def __init__(
        self,
        model,
        transform,
        dataset,
        atm_light,
        add_ext_haze,
        t_low,
        t_high,
        t_map_random_sampler,
        uint8_transform,
        cuda,
        prediction_dir,
    ):
        self.model = model
        self.transform = transform
        self.dataset = dataset
        self.atm_light = atm_light
        self.add_ext_haze = add_ext_haze
        self.t_low = t_low
        self.t_high = t_high
        self.random_sampler = t_map_random_sampler
        self.uint8_transform = uint8_transform
        self.cuda = cuda
        self.prediction_dir = prediction_dir

    def predict(self):
        dataset = torchvision.datasets.ImageFolder(self.dataset, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        iter = 0
        for input_data, _ in tqdm(dataloader, desc="Prediction"):
            if self.add_ext_haze:
                input_data, hazy_data, _ = generate_input(
                    input_data=input_data,
                    t_low=self.t_low,
                    t_high=self.t_high,
                    A=self.atm_light,
                    cuda=self.cuda,
                    uint8_transform=self.uint8_transform,
                    random_sampler=self.random_sampler,
                )
            else:
                input_data, hazy_data = feature_maps(
                    input=input_data,
                    cuda=self.cuda,
                )
            output_data = self.model(input_data)
            output_data = clear_image(output_data, hazy_data, self.atm_light)
            fname = tools.get_file_name(dataloader.dataset.samples[iter])
            iter += 1
            save_image(output_data[0], self.prediction_dir, fname)
