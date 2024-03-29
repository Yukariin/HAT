from math import exp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def rgb2ycbcr(x, y_only=False):
    if y_only:
        w = torch.tensor([[65.481], [128.553], [24.966]]).to(x)
        out = torch.matmul(x.permute(0, 2, 3, 1), w).permute(0, 3, 1, 2) + 16.0
    else:
        w = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(x)
        b = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(x)
        out = torch.matmul(x.permute(0, 2, 3, 1), w).permute(0, 3, 1, 2) + b
    
    out = out / 255.
    return out


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super().__init__()

        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze_(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, full=False):
    pad = window_size//2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    cs_map = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        cs = torch.mean(cs_map)
        return ret, cs

    return ret


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, y_channel=False):
        if y_channel:
            img1 = rgb2ycbcr(img1, y_only=True)
            img2 = rgb2ycbcr(img2, y_only=True)

        c = img1.size(1)

        if c == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, c)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = c

        return _ssim(img1, img2, window, self.window_size, c, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True, full=False, y_channel=False):
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'

    if y_channel:
        img1 = rgb2ycbcr(img1, y_only=True)
        img2 = rgb2ycbcr(img2, y_only=True)

    _, c, h, w = img1.size()

    real_size = min(window_size, h, w)
    window = create_window(real_size, c)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, real_size, c, size_average, full)


def msssim(img1, img2, window_size=11, size_average=True, y_channel=False):
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'

    if y_channel:
        img1 = rgb2ycbcr(img1, y_only=True)
        img2 = rgb2ycbcr(img2, y_only=True)

    weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    if img1.is_cuda:
        weights = weights.cuda(img1.get_device())
    weights = weights.type_as(img1)

    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size, size_average, full=True)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    input = torch.prod(pow1[:-1] * pow2[-1])
    return input


def psnr(input, target, y_channel=False, max_val=1.):
    assert input.shape == target.shape, f'Image shapes are different: {input.shape}, {target.shape}.'

    if y_channel:
        input = rgb2ycbcr(input, y_only=True)
        target = rgb2ycbcr(target, y_only=True)

    mse = F.mse_loss(input, target)
    psnr = 10. * torch.log10(max_val / mse)

    return psnr


# https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
def blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3]-1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2]-1, 8)

    horizontal_block_difference = ((im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1])**2).sum(3).sum(2).sum(1)
    vertical_block_difference = ((im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :])**2).sum(3).sum(2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(torch.arange(0,im.shape[3]-1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0,im.shape[2]-1), block_vertical_positions)

    horizontal_nonblock_difference = ((im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1])**2).sum(3).sum(2).sum(1)
    vertical_nonblock_difference = ((im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :])**2).sum(3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3]//block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2]//block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def psnrb(input, target, y_channel=False, max_val=1.):
    assert input.shape == target.shape, f'Image shapes are different: {input.shape}, {target.shape}.'

    if y_channel:
        input = rgb2ycbcr(input, y_only=True)
        target = rgb2ycbcr(target, y_only=True)

    total = 0
    for c in range(input.shape[1]):
        mse = F.mse_loss(input[:, c:c+1, :, :], target[:, c:c+1, :, :], reduction='none')
        bef = blocking_effect_factor(input[:, c:c+1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10. * torch.log10(max_val / (mse + bef))

    return total / input.shape[1]


# https://github.com/lusinlu/gradient-variance-loss/blob/main/gradient_variance_loss.py
class GradientVariance(nn.Module):
    def __init__(self, patch_size, cpu=False):
        super().__init__()

        self.patch_size = patch_size

        self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        if not cpu:
            self.kernel_x = self.kernel_x.cuda()
            self.kernel_y = self.kernel_y.cuda()

        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)
    
    def forward(self, input, target):
        gray_input = 0.2989 * input[:, 0:1, :, :] + 0.5870 * input[:, 1:2, :, :] + 0.1140 * input[:, 2:, :, :]
        gray_target = 0.2989 * target[:, 0:1, :, :] + 0.5870 * target[:, 1:2, :, :] + 0.1140 * target[:, 2:, :, :]

        gx_target = F.conv2d(gray_target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(gray_target, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(gray_input, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(gray_input, self.kernel_y, stride=1, padding=1)

        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        gradvar_loss = F.mse_loss(var_target_x, var_output_x) + F.mse_loss(var_target_y, var_output_y)
        return gradvar_loss