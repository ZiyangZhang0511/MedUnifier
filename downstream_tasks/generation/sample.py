import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np

from modeling.pixelsnail.pixelsnail import PixelSNAIL
from .utils import create_model, save_batch_images


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in range(size[0]):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--ckpt_full_model', type=str)
    parser.add_argument('--ckpt_top', type=str)
    parser.add_argument('--ckpt_bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    return args


def main():

    args = get_args()

    ###======== build models ========###
    model_full = create_model('full_model', args.ckpt_full_model, args.device)
    model_top = create_model('pixelsnail_top', args.ckpt_top, args.device)
    model_bottom = create_model('pixelsnail_bottom', args.ckpt_bottom, args.device)
    # model_full.eval()
    model_full.to(args.device)
    # model_top.eval()
    model_top.to(args.device)
    # model_bottom.eval()
    model_bottom.to(args.device)


    ###======== sample prior ========###

    for i, _ in enumerate(tqdm(range(0, args.num_samples, args.batch_size))):

        top_sample = sample_model(
            model_top,
            args.device,
            args.batch_size,
            [8, 8],
            args.temp
        )
        bottom_sample = sample_model(
            model_bottom,
            args.device,
            args.batch_size,
            [16, 16],
            args.temp,
            condition=top_sample,
        )

        ###======== decode code ========###
        decoded_samples = model_full.generate_images(top_sample, bottom_sample)
        decoded_samples = decoded_samples.clamp(0, 1)

        ###======== save generated images ========###
        save_batch_images(decoded_samples, args.save_path, start_idx=i)


if __name__ == "__main__":

    main()