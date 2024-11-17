import os
import pandas as pd
from PIL import Image
import re
import json
import random
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image

from transformers import AutoTokenizer

import numpy as np
import cv2
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from .utils import pre_caption, get_jpg_filepaths, generate_class_prompts, generate_combinations


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


class NewMimicCXRPretrainDataset(Dataset):
    def __init__(self, root_dir, mode="train", single_sentence=False):

        self.root_dir = root_dir
        self.single_sentence = single_sentence

        if mode == "train":
            self.image_dir = root_dir + "train_image_dir/"
            self.annotation_filepath = root_dir + "new_mimic_cxr_pairs_train.csv"
        elif mode == "val":
            self.image_dir = root_dir + "val_image_dir/"
            self.annotation_filepath = root_dir + "new_mimic_cxr_pairs_val.csv"
        elif mode == "test":
            self.image_dir = root_dir + "test_image_dir/"
            self.annotation_filepath = root_dir + "new_mimic_cxr_pairs_test.csv"

        
        self.annotations = pd.read_csv(self.annotation_filepath)
        self.image_relpaths = self.annotations["image_path"].tolist()
        self.texts = self.annotations["text_content"].tolist()

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.texts = self.annotations["text_content"].tolist()
        self.ip_gts = []
        for text in self.texts:
            multiple_sentences = []
            multiple_sentences = sent_tokenize(text)
            multiple_sentences_proc = [pre_caption(sentence, max_words=95) for sentence in multiple_sentences]
            self.ip_gts.append(multiple_sentences_proc)

    def __getitem__(self, idx):

        image_filename = self.annotations.iloc[idx, 3]
        image_filepath = self.image_dir + image_filename
        image = read_image(image_filepath, mode=ImageReadMode.RGB)
        image = self.transform(image)

        full_captions = self.annotations.iloc[idx, 4]

        if self.single_sentence:
            multiple_sentences = sent_tokenize(full_captions)
            multiple_sentences_proc = [pre_caption(sentence, max_words=95) for sentence in multiple_sentences]
            caption = random.choice(multiple_sentences_proc)

        else:
            caption = pre_caption(full_captions, max_words=95)

        return image, caption


    def __len__(self):
        return len(self.annotations)


class NewMimicCXRTestDataset(Dataset):

    def __init__(self, root_dir, data_ratio=1.0, single_sentence=False):

        self.root_dir = root_dir
        self.single_sentence = single_sentence

        self.image_dir = root_dir + "image_dir/"

        self.annotation_filepath = root_dir + "new_mimic_cxr_pairs_5x200.csv"
        self.annotations = pd.read_csv(self.annotation_filepath)

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.image_relpaths = self.annotations["image_path"].tolist()
        self.texts = self.annotations["text_content"].tolist()

        self.txt2img = {i: [] for i in range(len(self.image_relpaths))}
        self.img2txt = {i: [] for i in range(len(self.image_relpaths))}

        for img_id, image_relpath in enumerate(self.image_relpaths):
            self.img2txt[img_id] = []
            imgrelpath_cap_dict = self.get_captions_of_label(image_relpath)
            
            for imgrelpath, cap in imgrelpath_cap_dict.items():
                idx = self.image_relpaths.index(imgrelpath)
                self.img2txt[img_id].append(idx)
                self.txt2img[idx].append(img_id)

        self.ip_gts = []
        for text in self.texts:
            multiple_sentences = sent_tokenize(text)
            multiple_sentences_proc = [pre_caption(sentence, max_words=95) for sentence in multiple_sentences]
            self.ip_gts.append(multiple_sentences_proc)


    def get_captions_of_label(self, image_relpath):

        label = self.get_label(image_relpath)

        filtered_label_df = self.annotations[self.annotations["labels"] == label]
        image_relpaths_with_label = filtered_label_df["image_path"].tolist()
        img_cap_dict = {}
        for image_relpath in image_relpaths_with_label:
            caption = self.annotations[self.annotations["image_path"]==image_relpath]["text_content"].values[0]
            img_cap_dict[image_relpath] = caption
        return img_cap_dict


    def __getitem__(self, idx):

        ### get image ###
        image_relpath = self.annotations.loc[idx, "image_path"]
        img_path = os.path.join(self.image_dir, image_relpath)
        image = read_image(img_path, mode=ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)

        full_captions = self.annotations.loc[idx, "text_content"]

        if self.single_sentence:
            multiple_sentences = sent_tokenize(full_captions)
            multiple_sentences_proc = [pre_caption(sentence, max_words=95) for sentence in multiple_sentences]
            caption = random.choice(multiple_sentences_proc)
        else:
            caption = pre_caption(full_captions, max_words=95)

        return image, caption
    
    def __len__(self):
        return len(self.annotations)

    def get_label(self, image_relpath):
        label = self.annotations[self.annotations["image_path"] == image_relpath]["labels"].iloc[0]
        return label


class MimicZeroShotDataset(Dataset):

    def __init__(self, root_dir, class_prompts_dict, num_prompts_per_class=10):

        self.image_dir = root_dir + "image_dir/"
        self.annotation_filepath = root_dir + "new_mimic_cxr_pairs_5x200.csv"

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

        image_filename = self.annotations_df.iloc[idx, 3]
        image_filepath = os.path.join(self.image_dir + image_filename)
        image = read_image(image_filepath, mode=ImageReadMode.RGB)
        image = self.transform(image)

        class_name = self.annotations_df.iloc[idx, 5].strip("[]'\"")
        label = self.class2index[class_name]

        return image, label

    def get_class_name(self, image_filename):
        filtered_df = self.annotations_df[self.annotations_df["Filename"] == image_filename]

        non_zero_mask = filtered_df.drop('Filename', axis=1) != 0
        non_zero_column = filtered_df.columns[1:][non_zero_mask.values.argmax(axis=1)].tolist()
        class_name = non_zero_column[0]
        return class_name