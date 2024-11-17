from tqdm.auto import tqdm
import shutil
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import numpy as np

from lavis.models import load_model_and_preprocess, load_model

from . import metrics
from .constants import Constants
from datasets.mimic_cxr import NewMimicCXRPretrainDataset, NewMimicCXRTestDataset




def remove_directory(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Error occurred while removing the directory: {e}")

def remove_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error occurred while removing the file: {e}")

def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    images = torch.stack(images)
    return {"image": images, "text_input": texts}

def collate_fn_test(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    # labels = [item[2] for item in batch]
    images = torch.stack(images)

    return {"image": images, "text_input": texts}


def build_dataloader(dataset, batch_size=16, mode="train"):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode=="train" else False,
        drop_last=True if mode=="train" else False,
        collate_fn=collate_fn,
        num_workers=8,
    )
    return dataloader

def build_dataset(dataset_name, mode="train", single_sentence=False):
    
    if dataset_name == "new_mimic_cxr_full":
        dataset = NewMimicCXRPretrainDataset(
            root_dir=Constants.new_mimic_cxr_full_dir,
            mode=mode,
            single_sentence=single_sentence,
        )

    elif dataset_name == "new_mimic_cxr_5x200":
        dataset = NewMimicCXRTestDataset(
            root_dir=Constants.new_mimic_cxr_5x200_dir,
            single_sentence=single_sentence,
        )
    
    return dataset


