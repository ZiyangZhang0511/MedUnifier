import random
import os
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)
import numpy as np
import pandas as pd

from utilities import constants
from datasets.chexpert import CheXpertZeroShotDataset
from datasets.mimic_cxr import MimicZeroShotDataset
from datasets.rsna import RSNAZeroShotDataset, RSNAFinetuneDataset
from datasets.covidx import COVIDXFineTuneDataset
from datasets.siim import SIIMFinetuneDataset
from modeling.medunifier import medunifier


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
            requires_TIG=True,
        )
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)

    return model.float()


def build_zero_shot_dataset(dataset_name):
    if dataset_name == "chexpert_5x200":
        dataset = CheXpertZeroShotDataset(
            constants.Constants.chexpert_5x200_dir,
            constants.CHEXPERT_CLASS_PROMPTS,
            num_prompts_per_class=15,
        )

    elif dataset_name == "mimic_5x200":
        dataset = MimicZeroShotDataset(
            constants.Constants.new_mimic_cxr_5x200_dir,
            constants.CHEXPERT_CLASS_PROMPTS,
            num_prompts_per_class=15,
        )

    elif dataset_name == "rsna_1000":
        dataset = RSNAZeroShotDataset(constants.Constants.rsna_1000_dir, constants.RSNA_CLASS_PROMPTS)

    return dataset

def build_zero_shot_dataloader(dataset, batch_size=16):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )
    return dataloader

def build_fine_tune_dataset(dataset_name, mode="train", train_ratio=1.0):
        
    if dataset_name == "rsna_full":
        dataset = RSNAFinetuneDataset(constants.Constants.rsna_full_dir, mode=mode, train_ratio=train_ratio)
        
    elif dataset_name == "covidx":
        dataset = COVIDXFineTuneDataset(constants.Constants.covidx_dir, mode=mode, train_ratio=train_ratio)

    elif dataset_name == "siim":
        dataset = SIIMFinetuneDataset(constants.Constants.siim_dir, mode=mode, train_ratio=train_ratio)

    return dataset

    
def build_fine_tune_dataloader(dataset, batch_size=16, mode="train"):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if mode == "train" else False,
        drop_last=True if mode == "train" else False,
        num_workers=8,
    )
    return dataloader

@torch.no_grad()
def get_zero_shot_probilities(
    model,
    image_embeds,
    vit_feats,
    cls_prompts_mappting,
    requires_itm=False,
):

    class_similarities = []
    for cls_name, cls_prompts in cls_prompts_mappting.items():
        score_matrix_i2t = get_score_matrix(
            model, image_embeds, vit_feats, cls_prompts, requires_itm=requires_itm,
        )
        cls_similarity, _ = score_matrix_i2t.max(axis=1)
        class_similarities.append(cls_similarity)
    class_similarities = torch.stack(class_similarities, axis=1)
    class_similarities_df = pd.DataFrame(
        class_similarities.numpy(), columns=cls_prompts_mappting.keys()
    )


    ### normalize matrix by softmax to get probilities
    class_probs = F.softmax(class_similarities, dim=-1)
    class_probs_df = pd.DataFrame(
        class_probs.numpy(), columns=cls_prompts_mappting.keys()
    )

    return class_probs.numpy()


@torch.no_grad()
def get_score_matrix(
    model,
    image_embeds,
    vit_feats,
    cls_prompts,
    requires_itm=False,
):
    ###======== get prompts embeddings ========###
    text_input = model.tokenizer(
        cls_prompts,
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(image_embeds.device)
    text_feats = model.forward_text(text_input)
    text_embeds = F.normalize(model.text_proj(text_feats))
    text_ids = text_input.input_ids
    text_atts = text_input.attention_mask

    ###======== get i2t score matrix ========###
    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(image_embeds), len(cls_prompts)), -100.0
    ).to(image_embeds.device)

    for i, sims in enumerate(sims_matrix):
        if requires_itm:
            image_inputs = vit_feats[i].repeat(len(cls_prompts), 1, 1).to(image_embeds.device)
            score = model.compute_itm(
                image_inputs=image_inputs,
                text_ids=text_ids,
                text_atts=text_atts,
            ).float()
        else:
            score = 0.
        score_matrix_i2t[i] = score + sims

    return score_matrix_i2t.detach().cpu()


def multiclass_classification_metrics(probs, ground_truth):
    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(ground_truth, preds)
    try:
        auc = roc_auc_score(ground_truth, probs, multi_class='ovr')
    except ValueError:
        auc = None  # Handle case if AUC can't be computed

    f1 = f1_score(ground_truth, preds, average='macro')
    conf_matrix = confusion_matrix(ground_truth, preds)
    sensitivity, specificity, ppv, npv = [], [], [], []
    
    for i in range(conf_matrix.shape[0]):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        sens = TP / (TP + FN) if TP + FN > 0 else 0
        spec = TN / (TN + FP) if TN + FP > 0 else 0
        ppv_val = TP / (TP + FP) if TP + FP > 0 else 0
        npv_val = TN / (TN + FN) if TN + FN > 0 else 0

        sensitivity.append(sens)
        specificity.append(spec)
        ppv.append(ppv_val)
        npv.append(npv_val)

    sensitivity = np.mean(sensitivity)
    specificity = np.mean(specificity)
    ppv = np.mean(ppv)
    npv = np.mean(npv)

    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
    }

    return metrics


def binary_classification_metrics(probs, ground_truth):
    
    preds = (probs > 0.5).astype(int)

    accuracy = accuracy_score(ground_truth, preds)

    auc = roc_auc_score(ground_truth, probs)

    f1 = f1_score(ground_truth, preds)

    conf_matrix = confusion_matrix(ground_truth, preds)
    TN, FP, FN, TP = conf_matrix.ravel()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
    }

    return metrics

def binary_multiclass_classification_metrics(probs, ground_truth, threshold=0.5):
    preds = (probs >= threshold).astype(int)

    accuracy = []
    auc = []
    f1 = []
    sensitivity = []
    specificity = []
    ppv = []
    npv = []

    num_classes = ground_truth.shape[1]
    for i in range(num_classes):

        y_pred = preds[:, i]
        y_true = ground_truth[:, i]

        if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
            print(f"Class {i} does not occur in the test set.")
            continue

        acc = accuracy_score(y_true, y_pred)
        accuracy.append(acc)

        try:
            auc_score = roc_auc_score(y_true, probs[:, i])
        except ValueError:
            auc_score = np.nan
        auc.append(auc_score)

        f1_score_val = f1_score(y_true, y_pred, zero_division=0)
        f1.append(f1_score_val)

        conf_matrix = confusion_matrix(y_true, y_pred)

        if conf_matrix.shape == (2, 2):
            tn, fp, fn, tp = conf_matrix.ravel()
        elif conf_matrix.shape == (1, 1):
            if y_true[0] == 1:
                tp, tn, fp, fn = conf_matrix[0, 0], 0, 0, 0
            else:
                tn, tp, fp, fn = conf_matrix[0, 0], 0, 0, 0
        elif conf_matrix.shape == (1, 2):
            tn, fp = conf_matrix[0]
            fn, tp = 0, 0
        elif conf_matrix.shape == (2, 1):
            fn, tp = conf_matrix[:, 0]
            tn, fp = 0, 0


        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivity.append(sens)

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)

        ppv_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        ppv.append(ppv_val)

        npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0
        npv.append(npv_val)

    metrics = {
        'accuracy': np.mean(accuracy),
        'auc': np.nanmean(auc),
        'f1_score': np.mean(f1),
        'sensitivity': np.mean(sensitivity),
        'specificity': np.mean(specificity),
        'ppv': np.mean(ppv),
        'npv': np.mean(npv),
    }

    return metrics