import argparse
from tqdm import tqdm
import pickle
import os
import random

from accelerate import Accelerator

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import v2
from torchvision.utils import save_image

from pytorch_lightning import seed_everything

from modeling.medunifier import medunifier
from utilities.utils import build_dataset, build_dataloader, remove_directory

PRETRAIN_CHECKPOINTS_DIR = "/home/olg7848/p32335/my_research/MedUnifier/checkpoints/full_model/"


def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )

    parser.add_argument("--initial_lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_epochs", default=500, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--single_sentence", action="store_true")

    parser.add_argument("--cur_epoch", default=-1, type=int)
    parser.add_argument("--resume_pretraining", action="store_true")
    parser.add_argument("--restart_lr_scheduler", action="store_true")

    args = parser.parse_args()

    return args


def train_function(model, train_dataloader, val_dataloader, test_dataloader, args):
    
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, len(train_dataloader)*args.num_epochs, eta_min=1e-8)

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )

    if not args.resume_pretraining:
        cur_epoch = -1
        step = 0
        best_loss = 1e10
        accelerator.save_state(PRETRAIN_CHECKPOINTS_DIR + f"checkpoint_{cur_epoch}", safe_serialization=False)
    else: 
        cur_epoch = args.cur_epoch

        best_loss_filepath = PRETRAIN_CHECKPOINTS_DIR + "best_loss.pkl"
        with open(best_loss_filepath, 'rb') as file:
            best_loss = pickle.load(file)
        accelerator.print(f"current best loss: {best_loss}")

        step = (cur_epoch+1) * len(train_dataloader) * accelerator.num_processes 
        accelerator.load_state(PRETRAIN_CHECKPOINTS_DIR + f"checkpoint_{cur_epoch}", **{"strict":False})
    
    num_epochs = args.num_epochs

    if args.restart_lr_scheduler:
        optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.95), weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, len(train_dataloader)*args.num_epochs, eta_min=1e-8)
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    for epoch in range(cur_epoch+1, num_epochs):

        epoch_loss = 0
        epoch_loss_tig = 0
        epoch_loss_itc = 0
        epoch_loss_itm = 0
        epoch_loss_itg = 0
        model.train()

        accelerator.print("Training......")
        for i, batch in enumerate(tqdm(train_dataloader)):

            batch["image"] = batch["image"].to(accelerator.device)

            output = model(batch)
            
            loss = output["loss"]
            loss_vae = output["loss_tig"]
            loss_itc = output["loss_itc"]
            loss_itm = output["loss_itm"]
            loss_lm = output["loss_itg"]

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() / len(train_dataloader)
            epoch_loss_tig += loss_tig.item() / len(train_dataloader)
            epoch_loss_itc += loss_itc.item() / len(train_dataloader)
            epoch_loss_itm += loss_itm.item() / len(train_dataloader)
            epoch_loss_itg += loss_itg.item() / len(train_dataloader)
            step += 1 * accelerator.num_processes

        accelerator.print(f"Epoch {epoch}, current lr: {scheduler.get_last_lr()}, loss: {epoch_loss}, loss_tig: {epoch_loss_tig}, loss_itc: {epoch_loss_itc}, loss_itm: {epoch_loss_itm}, loss_itg: {epoch_loss_itg}")
        
        ###======== evaluate model on single process ========###
        accelerator.print("Validating......")
        model.eval()
        model_unwraped = accelerator.unwrap_model(model)
        model_unwraped.eval()
        metrics_dict = model_unwraped.evaluate(
            val_dataloader,
            step,
        )
        accelerator.print(metrics_dict)

        val_loss = metrics_dict["loss"]
        if val_loss < best_loss:
            best_loss = val_loss
            with open(PRETRAIN_CHECKPOINTS_DIR + "best_loss.pkl", 'wb') as file:
                pickle.dump(best_loss, file)
        
            accelerator.save_state(PRETRAIN_CHECKPOINTS_DIR + f"checkpoint_{epoch}_best", safe_serialization=False)
            accelerator.print(f"saved a better ckpt at {epoch}")

        elif accelerator.is_main_process: 
            accelerator.save_state(PRETRAIN_CHECKPOINTS_DIR + f"checkpoint_{epoch}", safe_serialization=False)
            accelerator.print(f"saved a ckpt at {epoch}")
        accelerator.wait_for_everyone()

        if epoch > 0 and accelerator.is_main_process:
            dir_path = PRETRAIN_CHECKPOINTS_DIR + f"checkpoint_{epoch-2}"
            if os.path.exists(dir_path):
                remove_directory(dir_path)
        accelerator.wait_for_everyone()

def main():

    args = get_args()

    ###======== get dataset ======###
    train_dataset = build_dataset("new_mimic_cxr_full", mode="train")
    val_dataset = build_dataset("new_mimic_cxr_full", mode="val")
    test_dataset = build_dataset("new_mimic_cxr_full", mode="test")

    train_dataloader = build_dataloader(train_dataset, batch_size=args.batch_size, mode="train")
    val_dataloader = build_dataloader(val_dataset, mode="val")
    test_dataloader = build_dataloader(test_dataset, mode="test")

    ###======== get model ========###
    model = medunifier.MedUnifier(
        max_txt_len=95,
        codebook_size=512,
    )

    ###======== train qformer and save checkpoint ========###
    train_function(model, train_dataloader, val_dataloader, test_dataloader, args)


if __name__ == "__main__":
    seed_everything(1)
    main()