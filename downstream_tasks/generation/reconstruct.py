import argparse
from tqdm import tqdm
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

import numpy as np

from modeling.pixelsnail.pixelsnail import PixelSNAIL
from .utils import create_model, LMDBDataset, save_batch_images

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ckpt_full_model', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--enc_ids_path", type=str)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    ###======== get dataset ========###
    dataset = LMDBDataset(path=args.enc_ids_path)
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    ###======== load model ========###
    model_full = create_model('full_model', args.ckpt_vqvae_qformer, args.device)
    model_full.to(args.device)
    # model_full.eval()

    ###======== decode ========###

    for i, (top, bottom, filename) in enumerate(tqdm(dataloader)):
        top = top.to(args.device)
        bottom = bottom.to(args.device)
        generated_images = model_full.generate_images(top, bottom)
        save_batch_images(generated_images, args.save_path, start_idx=i)
        


if __name__ == "__main__":
    main()