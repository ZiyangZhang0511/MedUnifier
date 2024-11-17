import re
import os
import random
import itertools

from utilities.constants import CHEXPERT_CLASS_PROMPTS


def pre_caption(caption, max_words=100):
    caption = re.sub(
        # r"([.!\"()*#:;~])",
        r"([!\"()*#:;~])",
        ' ',
        # caption.lower(),
        caption,
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        # caption.lower(),
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


def get_jpg_filepaths(directory):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files


def generate_class_prompts(class_prompts_dict, n: int = 10):

    prompts = {}
    for k, v in class_prompts_dict.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        prompts[k] = random.sample(cls_prompts, n)
    return prompts
