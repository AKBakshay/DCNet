import pdb
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.utils.data
import yaml
from torchvision import transforms
from torchvision.transforms.transforms import Resize

import src.common.tools as tools
import src.config.config as config
from src.model.nn.dcnet import DCNet
from src.test.predictor import Predictor
from src.test.tester import Tester
from src.train.trainer import Trainer


def train(cfg):

    # -------------------- data ------------------------

    training_data_transform = transforms.Compose(
        [
            transforms.RandomCrop(cfg["train"]["crop_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    validation_data_transform = transforms.Compose([transforms.ToTensor()])

    # -------------------- model -----------------------

    model = DCNet()
    model.initialize(cfg["basic"]["load_weight"], cfg["basic"]["cuda"])  # define

    # ------------------ training setup -------------------------

    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        alpha=cfg["train"]["alpha"],
        momentum=cfg["train"]["momentum"],
    )
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["train"]["scheduler_steps"], gamma=cfg["train"]["gamma"]
    )

    model.train()

    trainer = Trainer(
        model=model,
        cuda=cfg["basic"]["cuda"],
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=exp_lr_scheduler,
        train_crops=cfg["train"]["crops"],
        crop_size=cfg["train"]["crop_size"],
        epochs=cfg["train"]["epochs"],
        training_dataset_path=cfg["train"]["data_path"],
        validation_dataset_path=cfg["validate"]["data_path"],
        train_transform=training_data_transform,
        valid_transform=validation_data_transform,
        batch_size=cfg["train"]["batch_size"],
        t_low=cfg["env"]["transmission_map"]["low"],
        t_high=cfg["env"]["transmission_map"]["high"],
        atm_light=cfg["env"]["atm_light"],
        t_map_random_sampler=cfg["env"]["transmission_map"]["random_sampler"],
        uint8_transform=cfg["basic"]["uint8_transform"],
        save_path=cfg["output"]["weight"],
    )

    trainer.train()


def test(cfg):

    # -------------------- data ------------------------

    data_transform = transforms.Compose([transforms.ToTensor()])

    # -------------------- model -----------------------

    model = DCNet()
    model.initialize(cfg["basic"]["load_weight"], cfg["basic"]["cuda"])  # define

    # ------------------ test ---------------------------

    criterion = torch.nn.MSELoss(reduction="mean")

    model.eval()

    tester = Tester(
        model=model,
        cuda=cfg["basic"]["cuda"],
        criterion=criterion,
        test_dataset_path=cfg["test"]["data_path"],
        test_transform=data_transform,
        t_low=cfg["env"]["transmission_map"]["low"],
        t_high=cfg["env"]["transmission_map"]["high"],
        atm_light=cfg["env"]["atm_light"],
        random_sampler=cfg["env"]["transmission_map"]["random_sampler"],
        uint8_transform=cfg["basic"]["uint8_transform"],
    )

    tester.test()


def predict(cfg):

    # -------------------- data ------------------------

    data_transform = transforms.Compose([transforms.ToTensor()])

    # -------------------- model -----------------------
    model = DCNet()
    model.initialize(cfg["basic"]["load_weight"], cfg["basic"]["cuda"])  # define
    # -------------------- prediction ------------------
    model.eval()

    predictor = Predictor(
        model=model,
        transform=data_transform,
        dataset=cfg["predict"]["data_path"],
        atm_light=cfg["env"]["atm_light"],
        cuda=cfg["basic"]["cuda"],
    )

    predictor.predict()


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--training_dataset_path", default="data/Custom_dataset", type=str)
    # parser.add_argument("--validation_dataset_path", default="data/Valid", type=str)
    # parser.add_argument("--test_data_path", default="data/Valid", type=str)
    # parser.add_argument(
    #     "--prediction_data_path", default="data/DatasetForSomeMoreResults", type=str
    # )
    # parser.add_argument(
    #     "--model_weights", default="src/output/model_parameters/model.pt", type=str
    # )
    # parser.add_argument("--save_path", default="src/output/model_parameters/model.pt", type=str)
    # parser.add_argument("--train", action="store_true")
    # parser.add_argument("--test", action="store_true")
    # parser.add_argument("--available_clean_data", action="store_true")
    # parser.add_argument("--transmission_map_random_sampler", action="store_true")
    # parser.add_argument("--predict", action="store_true")
    # parser.add_argument(
    #     "--train_crops",
    #     type=int,
    #     default=15,
    #     help="number of crops per image for training",
    # )
    # parser.add_argument("--crop_size", type=int, default=20)
    # parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--cuda", action="store_true")
    # parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--learning_rate", type=float, default=1e-4)
    # parser.add_argument("--alpha", type=float, default=0.99)
    # parser.add_argument("--momentum", type=float, default=0.7)
    # parser.add_argument("--gamma", type=float, default=0.5)
    # parser.add_argument("--scheduler_steps", type=int, default=50)
    # parser.add_argument("--uint8_transform", action="store_true")
    # parser.add_argument("--add_haze", action="store_true")
    # parser.add_argument("--transmission_map_low", type=float, default=0.5)
    # parser.add_argument("--transmission_map_high", type=float, default=0.9)
    # parser.add_argument("--tmap_random", action="store_true")
    # parser.add_argument("--atm_light", type=float, default=0.9)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    with open(config.path["CONFIG_PATH"], "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    if args.train:
        train(cfg)

    if args.test:
        test(cfg)

    if args.predict:
        predict(cfg)
