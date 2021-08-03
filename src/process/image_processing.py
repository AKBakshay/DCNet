import copy
from pathlib import Path

import numpy as np
import torch
import torchvision
from matplotlib.pyplot import imshow, show
from PIL import Image
from scipy import ndimage

# def generate_input(input_data, t_low, t_high, A, cuda, uint8_transform, random_sampler):
#     if random_sampler:
#         t = np.random.uniform(t_low, t_high, input_data.shape[0])  # Transmission map
#     else:
#         t = np.ones(input_data.shape[0]) * t_high
#     X = copy.deepcopy(input_data)
#     X = add_haze(X, t, A)
#     X = transform_val_uint8(X, uint8_transform)
#     hazy_data = copy.deepcopy(X)
#     X = X.detach().numpy()

#     min_map = np.amin(X, axis=1)
#     V = np.amax(X, axis=1)  # Value channel
#     depth1 = min_map  # dark channel 1x1
#     depth3 = Depth(min_map, 3)  # dark channel 3x3
#     depth5 = Depth(min_map, 5)  # dark channel 5x5
#     depth7 = Depth(min_map, 7)  # dark channel 7x7
#     depth10 = Depth(min_map, 10)  # dark channel 10x10

#     Input_Tensor = concatenate_maps(
#         depth1, depth3, depth5, depth7, depth10, V
#     )  # [check] dimensions
#     Input_Tensor = torch.from_numpy(Input_Tensor)

#     device = torch.device("cuda" if cuda else "cpu")
#     Input_Tensor = Input_Tensor.double().to(device)
#     hazy_data = hazy_data.double().to(device)
#     G = input_data.double().to(device)

#     return Input_Tensor, hazy_data, G


def generate_input(input_data, t_low, t_high, A, cuda, uint8_transform, random_sampler):
    if random_sampler:
        t = np.random.uniform(t_low, t_high, input_data.shape[0])  # Transmission map
    else:
        t = np.ones(input_data.shape[0]) * (t_high + t_low) / 2.0
    device = torch.device("cuda" if cuda else "cpu")

    G = input_data.double().to(device)
    X = copy.deepcopy(input_data)
    X = add_haze(X, t, A)
    X = transform_val_uint8(X, uint8_transform)

    Input_Tensor, hazy_data = feature_maps(X, cuda)

    return Input_Tensor, hazy_data, G


def feature_maps(input, cuda):
    hazy_data = copy.deepcopy(input)
    input = input.detach().numpy()

    min_map = np.amin(input, axis=1)
    V = np.amax(input, axis=1)  # Value channel
    depth1 = min_map  # dark channel 1x1
    depth3 = Depth(min_map, 3)  # dark channel 3x3
    depth5 = Depth(min_map, 5)  # dark channel 5x5
    depth7 = Depth(min_map, 7)  # dark channel 7x7
    depth10 = Depth(min_map, 10)  # dark channel 10x10

    Input_Tensor = concatenate_maps(
        depth1, depth3, depth5, depth7, depth10, V
    )  # [check] dimensions
    Input_Tensor = torch.from_numpy(Input_Tensor)

    device = torch.device("cuda" if cuda else "cpu")
    Input_Tensor = Input_Tensor.double().to(device)
    hazy_data = hazy_data.double().to(device)

    return Input_Tensor, hazy_data


def add_haze(input, t, A):
    for i in range(input.shape[0]):
        input[i, :, :, :] = t[i] * input[i, :, :, :] + (1 - t[i]) * A
    return input


def transform_val_uint8(input, transform):
    if transform:
        return input.type(torch.uint8).type(torch.float32)
    return input


def Depth(input, window_size):
    X = np.ndarray(input.shape)
    for i in range(input.shape[0]):
        X[i, :, :] = ndimage.minimum_filter(input[i, :, :], size=window_size)
    return X


def concatenate_maps(x1, x2, x3, x4, x5, x6):
    return np.stack((x1, x2, x3, x4, x5, x6), axis=1)


def clear_image(t, input_data, A):
    transmission_map = t.clamp(min=0.1, max=1)  # [check] look at clamp min max
    output = depth2image(input_data, transmission_map, A)
    output = output.clamp(min=0, max=1)
    return output


def depth2image(input_data, transmission_map, A):
    output = input_data - A * (1 - transmission_map)
    output = torch.div(output, transmission_map)  # [check] if it correct
    output = output.clamp(min=0, max=1)
    return output


idx = 0


def save_image(img_tensor, save_dir, fname):
    # global idx
    # # img = torchvision.transforms.ToPILImage()((img_tensor.cpu() * 255).type(torch.uint8)).convert(
    # #     "RGB"
    # # )
    path = Path(save_dir)
    img = torchvision.transforms.ToPILImage()((img_tensor.cpu() * 255).type(torch.uint8))
    img.save(path / fname)
    # idx += 1
    # imshow(img)
    # show(img)


def show_image_grey(img_tensor):
    global idx
    img = torchvision.transforms.ToPILImage()((img_tensor.cpu() * 255).type(torch.uint8))
    img.save("data/results/tmp{}.png".format(idx))
    idx += 1
