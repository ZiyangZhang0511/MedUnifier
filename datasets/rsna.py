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


class RSNAZeroShotDataset(Dataset):

    def __init__(self, root_dir, class_prompts_dict, num_prompts_per_class=10):

        self.image_dir = root_dir + "testing_1000/"
        self.annotation_filepath = root_dir + "RSNA_pneumonia_self_labels_1000_with_name.csv"

        self.class2index = {
            "Pneumonia": 0,
            "Normal": 1,
        }

        self.annotations_df = pd.read_csv(self.annotation_filepath)

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.cls_prompts_mappting = generate_class_prompts(class_prompts_dict, num_prompts_per_class)

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):

        image_filename = self.annotations_df.iloc[idx, 0]
        image_filepath = os.path.join(self.image_dir + image_filename)
        image = read_image(image_filepath, mode=ImageReadMode.RGB)
        image = self.transform(image)

        class_name = self.get_class_name(image_filename)
        label = self.class2index[class_name]

        return image, label

    def get_class_name(self, image_filename):
        filtered_df = self.annotations_df[self.annotations_df["Filename"] == image_filename]

        non_zero_mask = filtered_df.drop('Filename', axis=1) != 0
        non_zero_column = filtered_df.columns[1:][non_zero_mask.values.argmax(axis=1)].tolist()
        class_name = non_zero_column[0]
        return class_name


class RSNAFinetuneDataset(Dataset):
    def __init__(self, root_dir, mode="train", train_ratio=1.0):
        self.train_ratio = train_ratio
        self.image_dir = root_dir + "all_images_dir/"
        self.annotation_filepath = root_dir + "annotations.csv"

        self.annotations_df = pd.read_csv(self.annotation_filepath)
        self.annotations_df = self.annotations_df[self.annotations_df["split"] == mode]

        if mode == "train" and train_ratio != 1.0:
            _, self.annotations_df = train_test_split(self.annotations_df, test_size=self.train_ratio, stratify=self.annotations_df['Target'])

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.class2index = {
            "Pneumonia": 0,
        }

        num_pos = (self.annotations_df["Target"] == 1.0).sum()
        num_neg = len(self.annotations_df) - num_pos
        pos_weight = num_neg / num_pos
        # self.pos_weights = list(self.class2pos_weight.values())
        self.pos_weights = torch.tensor(pos_weight)

    def __getitem__(self, idx):
        image_filename = self.annotations_df.iloc[idx, 0] + ".jpg"
        # image_filename = self.patientIds[idx] + ".jpg"
        image_filepath = os.path.join(self.image_dir + image_filename)
        image = read_image(image_filepath, mode=ImageReadMode.RGB)
        image = self.transform(image)

        # target = self.targets[idx]
        target = self.annotations_df.iloc[idx, 5]

        return image, torch.tensor(target, dtype=torch.float32).unsqueeze(0)


    def __len__(self):
        return len(self.annotations_df)


