import os
from PIL import Image
import re
import json
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image

from transformers import AutoTokenizer

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from .utils import pre_caption, get_jpg_filepaths, generate_class_prompts


class COVIDXFineTuneDataset(Dataset):
    
    def __init__(self, root_dir, mode="train", train_ratio=1.0):
        if mode == "train":
            self.image_dir = root_dir + "train/train/"
            self.annotation_filepath = root_dir + "train.csv"
        elif mode == "val":
            self.image_dir = root_dir + "train/train/"
            self.annotation_filepath = root_dir + "val.csv"
        elif mode == "test":
            self.image_dir = root_dir + "test/"
            self.annotation_filepath = root_dir + "test.csv"

        self.annotations_df = pd.read_csv(self.annotation_filepath, index_col=0)
        if mode == "train" and train_ratio != 1.0:
            self.annotations_df = self.annotations_df.sample(frac=train_ratio).reset_index(drop=True)

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.class2index = {
            "COVID-19": 0,
            "normal": 1,
            "pneumonia": 2,
        }


    def __getitem__(self, idx):

        image_filename = self.annotations_df.iloc[idx, 0]
        image_filepath = os.path.join(self.image_dir + image_filename)
        image = read_image(image_filepath, mode=ImageReadMode.RGB)
        image = self.transform(image)

        class_name = self.annotations_df.iloc[idx, 1]
        label = self.class2index[class_name]
        
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.annotations_df)
