import time

import src.common.tools as tools
import torch
import torchvision
from src.common.tools import format_time
from src.metrics.metrics import Metrics
from src.process.image_processing import (
    clear_image,
    generate_input,
    show_image,
    show_image_grey,
)
from torch.nn.functional import group_norm
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        cuda,
        criterion,
        optimizer,
        lr_scheduler,
        train_crops,
        crop_size,
        epochs,
        training_dataset_path,
        validation_dataset_path,
        train_transform,
        valid_transform,
        batch_size,
        t_low,
        t_high,
        atm_light,
        t_map_random_sampler,
        uint8_transform,
        save_path,
    ):
        self.model = model
        self.cuda = cuda
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_crops = train_crops
        self.crop_size = crop_size
        self.epochs = epochs
        self.training_dataset_path = training_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.batch_size = batch_size
        self.t_low = t_low
        self.t_high = t_high
        self.atm_light = atm_light
        self.random_sampler = t_map_random_sampler
        self.uint8_transform = uint8_transform
        self.save_path = save_path
        self.metrics = Metrics()

    def train(self):
        start_time = time.time()
        total_loss = 0
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.epochs))
            print("-" * 89)
            epoch_loss = self.training_epoch()
            print("Training loss: {}".format(epoch_loss))
            self.validation_epoch()
            self.metrics.show()
            total_loss += epoch_loss / self.epochs
        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Training complete in {:.0f}hr {:.0f}m {:.0f}s".format(*(format_time(time_elapsed))))
        torch.save(
            self.model.state_dict(),
            tools.get_save_dir(
                self.save_path,
            ),
        )

    def training_epoch(self):
        # self.metrics.reset()
        epoch_loss = 0.0
        for crop_no in tqdm(range(self.train_crops), desc="Training"):
            dataset = torchvision.datasets.ImageFolder(
                self.training_dataset_path, transform=self.train_transform
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False
            )
            running_loss = 0.0

            for input_data, _ in dataloader:
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(False):
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
                # show_image(output_data[0])
                # show_image(ground_truth[0])
                loss = self.criterion(output_data, ground_truth)
                # self.metrics.add(ground_truth, output_data)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * input_data.size()[0]
            epoch_loss += running_loss / len(dataset)
            self.lr_scheduler.step()
        return epoch_loss

    def validation_epoch(self):
        self.metrics.reset()

        dataset = torchvision.datasets.ImageFolder(
            self.validation_dataset_path, transform=self.valid_transform
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for input_data, _ in tqdm(dataloader, desc="Validating"):
            input_data, hazy_data, ground_truth = generate_input(
                input_data=input_data,
                t_low=self.t_low,
                t_high=self.t_high,
                A=self.atm_light,
                cuda=self.cuda,
                uint8_transform=self.uint8_transform,
                random_sampler=False,
            )

            output_data = self.model(input_data)
            output_data = clear_image(output_data, hazy_data, self.atm_light)
            # show_image(output_data[0])
            # show_image(ground_truth[0])
            self.metrics.add(ground_truth, output_data)

    # def validation_epoch(self):
    #     self.metrics.reset()
    #     dataset = torchvision.datasets.ImageFolder(
    #         self.validation_dataset_path, transform=self.valid_transform
    #     )
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset, batch_size=self.batch_size, shuffle=False
    #     )

    #     for input_data, _ in tqdm(dataloader, desc="Validating"):
    #         input_data, hazy_data, ground_truth = generate_input(
    #             input_data=input_data,
    #             t_low=self.t_low,
    #             t_high=self.t_high,
    #             A=self.atm_light,
    #             cuda=self.cuda,
    #             uint8_transform=self.uint8_transform,
    #         )

    #         output_data = self.model(input_data)
    #         output_data = clear_image(output_data, hazy_data, self.atm_light)
    #         # show_image(output_data[0])
    #         # show_image(ground_truth[0])
    #         self.metrics.add(ground_truth, output_data)
