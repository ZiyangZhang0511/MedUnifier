import random
import os
from tqdm.auto import tqdm
from collections import namedtuple
import pickle
import lmdb

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from PIL import Image

from utilities import constants
from datasets.mimic_cxr import NewMimicCXRPretrainDataset
from modeling.medunifier import medunifier
from modeling.pixelsnail.pixelsnail import PixelSNAIL

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])



def save_batch_images(tensor_batch, save_path, start_idx=0, prefix="image", format="png"):

    os.makedirs(save_path, exist_ok=True)
    N = tensor_batch.shape[0]

    tensor_batch = torch.clamp(tensor_batch, 0, 1)
    to_pil = transforms.ToPILImage()
    for idx, image_tensor in enumerate(tensor_batch):
        image = to_pil(image_tensor.cpu())
        filename = f"{prefix}_{start_idx*N+idx}.{format}"
        image.save(os.path.join(save_path, filename))



def create_model(model_type:str, ckpt_path:str, device="cuda"):
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    
    if model_type == "full_model":
        model = medunifier.MedUnifier(
            max_txt_len=95,
            codebook_size=512,
        )
        model.load_state_dict(checkpoint)
    
    elif model_type == "pixelsnail_top":
        model = PixelSNAIL(
            shape=[8, 8],
            n_class=512,
            channel=256,
            kernel_size=5,
            n_block=4,
            n_res_block=4,
            res_channel=256,
            dropout=0.1,
            n_out_res_block=0,
        )
        model.load_state_dict(checkpoint["model"])

    elif model_type == 'pixelsnail_bottom':
        model = PixelSNAIL(
            shape=[16, 16],
            n_class=512,
            channel=256,
            kernel_size=5,
            n_block=4,
            n_res_block=4,
            res_channel=256,
            attention=False,
            dropout=0.1,
            n_cond_res_block=3,
            cond_res_channel=256,
        )
        model.load_state_dict(checkpoint["model"])

    return model.float()


def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
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
            root_dir=constants.Constants.new_mimic_cxr_full_dir,
            mode=mode,
        )

    return dataset


class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename

