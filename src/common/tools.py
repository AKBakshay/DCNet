import datetime
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


def concatenate_maps(crop_size, x1, x2, x3, x4, x5, x6):
    result = np.empty((x1.shape[0], 6, crop_size, crop_size))
    result[:, 0, :, :] = x1
    result[:, 1, :, :] = x2
    result[:, 2, :, :] = x3
    result[:, 3, :, :] = x4
    result[:, 4, :, :] = x5
    result[:, 5, :, :] = x6
    return result


def format_time(time):
    return time // 3600, (time % 3600) // 60, time % 60


def get_images(data_path):
    files = []
    if os.path.isfile(data_path) and valid_image_format(data_path):
        files.append(data_path)
    elif os.path.isdir(data_path):
        files = os.listdir(data_path)
        files = [file for file in files if valid_image_format(file)]
    return files


def valid_image_format(data_path):
    valid_patterns = ["\w*.png", "\w*.jpg", "\w*.jpeg"]  # valid string patterns
    if not os.path.isfile(data_path):
        return False
    for pattern in valid_patterns:
        if re.match(pattern, data_path):
            return True
    return False


def avg(lst):
    return sum(lst) / len(lst)


def get_save_path(save_path):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d-%H%M%S")
    path = Path(save_path)
    return path / ("model_" + st)


def get_save_dir(path, save_type):
    fname = "model.pt"
    path = path / save_type
    if not os.path.isdir(path):
        os.makedirs(path)
    return path / fname


def save_best_performing_model(validation_metrics, best_metrics, model, save_path):
    if validation_metrics[1] > best_metrics[1]:  # ssim comparison
        torch.save(
            model.state_dict(),
            get_save_dir(save_path, "best_performing"),
        )
        print("saved model parameters")


def max(x, y):
    if x[1] >= y[1]:
        return x
    return y


def toTensor(files):
    result = []
    transform = transforms.ToTensor()
    for file in files:
        result.append(get_tensor(file, transform))
    return result


def get_tensor(file, transform):
    img = Image.open(file)
    result = transform(img)
    return result.unsqueeze(0)


def get_file_name(file):
    path = Path(file[0])
    return path.name
