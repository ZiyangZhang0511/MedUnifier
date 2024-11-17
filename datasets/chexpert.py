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

from .utils import pre_caption, get_jpg_filepaths, generate_class_prompts

class CheXpertZeroShotDataset(Dataset):

    def __init__(self, root_dir, class_prompts_dict, num_prompts_per_class=10):

        self.image_dir = root_dir + "testing_5x200/"
        self.annotation_filepath = root_dir + "chexpert_5x200_self_processed.csv"

        self.class2index = {
            "Atelectasis": 0,
            "Cardiomegaly": 1,
            "Pneumonia": 2,
            "Edema": 3,
            "Pleural Effusion": 4,
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
