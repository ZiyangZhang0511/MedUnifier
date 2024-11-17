import argparse
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from pytorch_lightning import seed_everything

from .utils import (
    build_zero_shot_dataset,
    build_zero_shot_dataloader, 
    create_model,
    get_zero_shot_probilities,
    binary_classification_metrics,
    multiclass_classification_metrics,
)


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, choices=["full_model", "incomplete_model"], required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, choices=["chexpert_5x200", "mimic_5x200", "rsna_1000"], required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args

def main():

    args = get_args()
    
    ###======== get dataset ========###
    dataset = build_zero_shot_dataset(args.dataset_name)
    dataloader = build_zero_shot_dataloader(dataset)
    # return

    ###======== load pretrained model ========###
    device = args.device
    model = create_model(args.model_type, args.ckpt_path)
    model.to(device)

    ###======== loop through dataloader ========###
    model.eval()
    image_embeds = []
    vit_feats = []
    image_labels = []
    for batch in dataloader:
        images = batch[0].to(device)
        labels = batch[1]
        with torch.no_grad():
            image_feat, vit_feat = model.forward_image(images)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)
        image_embeds.append(image_embed)
        vit_feats.append(vit_feat.cpu())
        image_labels.append(labels)

    image_embeds = torch.cat(image_embeds)
    vit_feats = torch.cat(vit_feats)
    image_labels = torch.cat(image_labels)

    ###======== calculate cls probs ========###
    class_probs = get_zero_shot_probilities(
        model,
        image_embeds,
        vit_feats,
        dataloader.dataset.cls_prompts_mappting,
        requires_itm=False,
    )

    ###======== evaluate performance ========###
    if args.dataset_name in ["chexpert_5x200", "mimic_5x200"]:
        metrics_dict = multiclass_classification_metrics(class_probs, image_labels.numpy())
        
    elif args.dataset_name in ["rsna_1000"]:
        metrics_dict = binary_classification_metrics(class_probs[:, 1], image_labels.numpy())
        
    print(metrics_dict)


if __name__ == "__main__":

    seed_everything(1)
    main()

    