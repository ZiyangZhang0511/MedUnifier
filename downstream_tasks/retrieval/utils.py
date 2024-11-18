import random
import os
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from utilities import constants
from datasets.mimic_cxr import NewMimicCXRTestDataset, NewMimicCXRPretrainDataset
from modeling.medunifier import medunifier


def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    images = torch.stack(images)
    return {"image": images, "text_input": texts}


def create_model(model_type:str, ckpt_path:str):
    if model_type == "full_model":
        model = medunifier.MedUnifier(
            max_txt_len=95,
            codebook_size=512,
        )
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)
        
    elif model_type == "incomplete_model":
        model = medunifier.MedUnifier(
            max_txt_len=95,
            codebook_size=512,
            requires_TIG=False,
        )
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)

    return model.float()



def build_retrieval_dataset(dataset_name, mode="test", single_sentence=False):

    if dataset_name == "new_mimic_cxr_5x200":
        dataset = NewMimicCXRTestDataset(
            root_dir=constants.Constants.new_mimic_cxr_5x200_dir,
            single_sentence=single_sentence,
        )

    elif dataset_name == "new_mimic_cxr_full":
        dataset = NewMimicCXRPretrainDataset(
            root_dir=constants.Constants.new_mimic_cxr_full_dir,
            mode=mode,
            single_sentence=single_sentence,
        )

    return dataset


def build_retrieval_dataloader(dataset, batch_size=16):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    return dataloader