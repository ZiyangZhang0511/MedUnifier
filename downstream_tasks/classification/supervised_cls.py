import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import seed_everything

from .utils import (
    build_fine_tune_dataset,
    build_fine_tune_dataloader, 
    create_model,
    get_zero_shot_probilities,
    binary_classification_metrics,
    multiclass_classification_metrics,
    binary_multiclass_classification_metrics,
)
from .classifier import Classifier


CHECKPOINTS_DIR = "./checkpoints/supervised_cls/"


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, choices=["full_model", "incomplete_model"], required=True)
    parser.add_argument("--pretrained_ckpt_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, choices=["rsna_full", "covidx", "siim"] , required=True)
    
    parser.add_argument("--initial_lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--train_ratio", default=1.0, type=float)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args

def main():

    args = get_args()
    
    ###======== get dataset ========###
    train_dataset = build_fine_tune_dataset(args.dataset_name, mode="train", train_ratio=args.train_ratio)
    val_dataset = build_fine_tune_dataset(args.dataset_name, mode="val")
    test_dataset = build_fine_tune_dataset(args.dataset_name, mode="test")

    train_dataloader = build_fine_tune_dataloader(train_dataset, args.batch_size, mode="train")
    val_dataloader = build_fine_tune_dataloader(val_dataset, mode="val")
    test_dataloader = build_fine_tune_dataloader(test_dataset, mode="test")


    ###======== creat classification model ========###
    device = args.device
    pretrained_model_config = {
        "model_type": args.model_type,
        "ckpt_path": args.pretrained_ckpt_path,
    }
    num_class = len(train_dataset.class2index)
    model = Classifier(pretrained_model_config, num_class)
    model.to(device)

    ###======== training loop ========###
    num_epochs = args.num_epochs
    best_loss = 1e10
    best_auc = 1e-10

    if args.dataset_name in ["rsna_full", "siim"]:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, len(train_dataloader)*args.num_epochs, eta_min=1e-8)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        print("Training......")
        for batch in tqdm(train_dataloader):
            images = batch[0].to(device)
            labels = batch[1].to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            epoch_loss += loss.item() / len(train_dataloader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch}: train loss: {epoch_loss}")

        print("Validating......")
        model.eval()
        metrics_dict = model.evaluate(val_dataloader, args.dataset_name, device=device)
        print(metrics_dict)
        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = CHECKPOINTS_DIR + f"{args.model_type}_{args.dataset_name}_{args.train_ratio}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print("saved a better ckpt successfully.")
        print(f"best val loss: {best_loss}")


    print("Testing......")
    ckpt_path = CHECKPOINTS_DIR + f"{args.model_type}_{args.dataset_name}_{args.train_ratio}.pth"
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    print(model.evaluate(test_dataloader, args.dataset_name, device=device))

            

if __name__ == "__main__":

    seed_everything(1)
    main()