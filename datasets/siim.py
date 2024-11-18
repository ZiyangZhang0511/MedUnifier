import os
from PIL import Image
import re
import json
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image

from transformers import AutoTokenizer

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split


class SIIMFinetuneDataset(Dataset):

    def __init__(self, root_dir, mode="train", train_ratio=1.0):
        self.train_ratio = train_ratio
        self.image_dir = root_dir + "image_dir/"
        self.annotation_filepath = root_dir + "annotations.csv"

        annotations_df = pd.read_csv(self.annotation_filepath, index_col=0)

        train_val_df, test_df = train_test_split(annotations_df, test_size=0.1, stratify=annotations_df["target"], random_state=5)
        train_df, val_df = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df["target"], random_state=5)

        if mode == "train":
            self.annotations_df = train_df
            if train_ratio != 1.0:
                _, self.annotations_df = train_test_split(self.annotations_df, test_size=self.train_ratio, stratify=self.annotations_df['target'], random_state=5)
        elif mode == "val":
            self.annotations_df = val_df
        elif mode == "test":
            self.annotations_df = test_df

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.class2index = {
            "Normal": 0
        }

    def __getitem__(self, idx):
        image_filename = self.annotations_df.iloc[idx, 0] + ".png"
        # image_filename = self.patientIds[idx] + ".jpg"
        image_filepath = os.path.join(self.image_dir + image_filename)
        image = read_image(image_filepath, mode=ImageReadMode.RGB)
        # return
        image = self.transform(image)

        target = self.annotations_df.iloc[idx, 2]

        return image, torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        
    def __len__(self):
        return len(self.annotations_df)
