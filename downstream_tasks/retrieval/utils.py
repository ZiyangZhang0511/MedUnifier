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
from datasets.open_i import OpenIPretrainDataset
from modeling.blip2_models import (
    blip2_qformer,
    blip2_qformer_3d,
    blip2_qformer_vqvae,
)

def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    images = torch.stack(images)
    return {"image": images, "text_input": texts}

def create_model(model_type:str, ckpt_path:str):
    if model_type == "vqvae_qformer":
        model = blip2_qformer_vqvae.Blip2QformerVQvae(
            vit_model="eva_clip_g",
            freeze_vit=True,
            max_txt_len=95,
            mode="2D",
            codebook_size=512,
            vit_precision="fp16",
        )
        checkpoint = torch.load(ckpt_path)
        # print(model.state_dict()["vqvae.downsample_zq.weight"].max())
        model.load_state_dict(checkpoint)
        # print(model.state_dict()["vqvae.downsample_zq.weight"].max())
        
    elif model_type == "plain_qformer":
        model = blip2_qformer.Blip2Qformer(
            vit_model="eva_clip_g",
            freeze_vit=True,
            max_txt_len=95,
        )
        checkpoint = torch.load(ckpt_path)
        # print(model.state_dict()["vqvae.downsample_zq.weight"].max())
        model.load_state_dict(checkpoint)
        # print(model.state_dict()["vqvae.downsample_zq.weight"].max())

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

    elif dataset_name == "open_i":
        if mode == "train":
            dataset = OpenIPretrainDataset(root_dir=constants.Constants.open_i_dir, mode="train", single_sentence=single_sentence)
        elif mode == "val":
            dataset = OpenIPretrainDataset(root_dir=constants.Constants.open_i_dir, mode="val", single_sentence=single_sentence)
        elif mode == "test":
            dataset = OpenIPretrainDataset(root_dir=constants.Constants.open_i_dir, mode="test", single_sentence=single_sentence)

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