import numpy as np
from nose.util import src
from numpy.core.fromnumeric import mean
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
from src.common.tools import avg


class Metrics:
    def __init__(self):
        self.epoch_loss = []
        self.epoch_ssim = []
        self.epoch_psnr = []

    def reset(self):
        self.epoch_loss = []
        self.epoch_ssim = []
        self.epoch_psnr = []

    # def add(self, src_data, output_data):
    #     self.evaluate_loss(src_data.detach().cpu().numpy(), output_data.detach().cpu().numpy())
    #     self.evaluate_ssim(src_data.detach().cpu().numpy(), output_data.detach().cpu().numpy())
    #     self.evaluate_psnr(src_data.detach().cpu().numpy(), output_data.detach().cpu().numpy())

    def add(self, src_data, output_data):
        src_data = src_data.detach().cpu().numpy()
        output_data = output_data.detach().cpu().numpy()
        src_data = np.moveaxis(src_data, 1, -1)
        output_data = np.moveaxis(output_data, 1, -1)
        self.evaluate_loss(src_data, output_data)
        self.evaluate_ssim(src_data, output_data)
        self.evaluate_psnr(src_data, output_data)

    def show(self):
        print("MSE: {}".format(avg(self.epoch_loss)))
        print("SSIM: {}".format(avg(self.epoch_ssim)))
        print("PSNR: {}".format(avg(self.epoch_psnr)))

    def evaluate_loss(self, src_data, output_data):
        batch_size = src_data.shape[0]
        for i in range(batch_size):
            self.epoch_loss.append(np.square(src_data[i] - output_data[i]).mean())

    def evaluate_ssim(self, src_data, output_data):
        batch_size = src_data.shape[0]
        for i in range(batch_size):
            self.epoch_ssim.append(ssim(src_data[i], output_data[i], multichannel=True))

    def evaluate_psnr(self, src_data, output_data):
        batch_size = src_data.shape[0]
        for i in range(batch_size):
            self.epoch_psnr.append(
                psnr(
                    src_data[i],
                    output_data[i],
                )
            )
