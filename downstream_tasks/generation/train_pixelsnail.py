import argparse
from tqdm import tqdm
from collections import namedtuple
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import numpy as np

from modeling.pixelsnail.pixelsnail import PixelSNAIL
from .utils import LMDBDataset
from utilities.utils import remove_file

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])

CHECKPOINTS_DIR = "./checkpoints/pixelsnail/pixelsnail_327/"


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument("--enc_ids_path", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--cur_epoch", default=-1, type=int)
    parser.add_argument("--resume_pretraining", action="store_true")

    args = parser.parse_args()

    return args


def train_function(model, dataloader, args):

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(dataloader)*args.num_epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()

    if not args.resume_pretraining:
        cur_epoch = -1
        torch.save(
            {'model': model.state_dict(), 'args': args},
            f'checkpoints/pixelsnail/pixelsnail_327/pixelsnail_{args.hier}_{str(cur_epoch).zfill(3)}.pth',
        )
    else: 
        cur_epoch = args.cur_epoch
        checkpoint = torch.load(CHECKPOINTS_DIR + f"pixelsnail_{args.hier}_{str(cur_epoch).zfill(3)}.pth")
        model.load_state_dict(checkpoint["model"])

    for epoch in range(cur_epoch+1, args.num_epochs):

        epoch_loss = 0
        epoch_accuracy = 0

        for i, (top, bottom, filename) in enumerate(tqdm(dataloader)):

            top = top.to(args.device)

            if args.hier == 'top':
                target = top
                out, _ = model(top)

            elif args.hier == 'bottom':
                bottom = bottom.to(args.device)
                target = bottom
                out, _ = model(bottom, condition=top)

            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
                
            optimizer.step()
            scheduler.step()

            _, pred = out.max(1)
            correct = (pred == target).float()
            accuracy = correct.sum() / target.numel()
            lr = optimizer.param_groups[0]['lr']

            epoch_loss += loss.item() / len(dataloader)
            epoch_accuracy += accuracy / len(dataloader)

        print(f"Epoch {epoch}, lr: {scheduler.get_last_lr()}, loss: {epoch_loss}, accuracy: {epoch_accuracy}")
        torch.save(
            {'model': model.state_dict(), 'args': args},
            f'checkpoints/pixelsnail/pixelsnail_327/pixelsnail_{args.hier}_{str(epoch).zfill(3)}.pth',
        )

        if epoch > 0:
            file_path = f'checkpoints/pixelsnail/pixelsnail_327/pixelsnail_{args.hier}_{str(epoch-4).zfill(3)}.pth'
            if os.path.exists(file_path):
                remove_file(file_path)



def main():

    args = get_args()

    ###======== build dataloader ========###
    dataset = LMDBDataset(path=args.enc_ids_path)
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    ###======== create pixelsnail model ========###
    if args.hier == 'top':
        model = PixelSNAIL(
            [8, 8],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [16, 16],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    model = model.to(args.device)
    
    ###======== train and save ========###
    train_function(model, dataloader, args)
    



if __name__ == "__main__":

    main()

