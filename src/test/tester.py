import time

import torch
import torchvision
from src.common.tools import format_time
from src.metrics.metrics import Metrics
from src.process.image_processing import clear_image, generate_input
from tqdm import tqdm


class Tester:
    def __init__(
        self,
        model,
        cuda,
        criterion,
        test_dataset_path,
        test_transform,
        t_low,
        t_high,
        atm_light,
        random_sampler,
        uint8_transform,
    ):
        self.model = model
        self.cuda = cuda
        self.criterion = criterion
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        self.t_low = t_low
        self.t_high = t_high
        self.atm_light = atm_light
        self.random_sampler = random_sampler
        self.uint8_transform = uint8_transform
        self.metrics = Metrics()

    def test(self):
        start_time = time.time()
        self.test_evaluate()
        self.metrics.show()
        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Test complete in {:.0f}hr {:.0f}m {:.0f}s".format(*(format_time(time_elapsed))))

    def test_evaluate(self):
        self.metrics.reset()

        dataset = torchvision.datasets.ImageFolder(
            self.test_dataset_path, transform=self.test_transform
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for input_data, _ in tqdm(dataloader, desc="Test"):
            input_data, hazy_data, ground_truth = generate_input(
                input_data=input_data,
                t_low=self.t_low,
                t_high=self.t_high,
                A=self.atm_light,
                cuda=self.cuda,
                uint8_transform=self.uint8_transform,
                random_sampler=self.random_sampler,
            )

            output_data = self.model(input_data)
            output_data = clear_image(output_data, hazy_data, self.atm_light)
            self.metrics.add(ground_truth, output_data)
