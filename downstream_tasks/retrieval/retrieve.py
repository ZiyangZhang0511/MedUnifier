import argparse
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from pytorch_lightning import seed_everything

from modeling.blip2_models import (
    blip2_qformer,
    blip2_qformer_3d,
    blip2_qformer_vqvae,
)

from .utils import build_retrieval_dataset, build_retrieval_dataloader, create_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, choices=["incomplete_model", "full_model"], required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, choices=["new_mimic_cxr_5x200", "new_mimic_cxr_full"], required=True)
    parser.add_argument("--single_sentence", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    
    ###======== get dataset ========###
    dataset = build_retrieval_dataset(args.dataset_name, mode="test", single_sentence=args.single_sentence)
    dataloader = build_retrieval_dataloader(dataset)
    # return

    ###======== load pretrained model ========###
    device = args.device
    model = create_model(args.model_type, args.ckpt_path)
    model.to(device)

    ###======== loop through dataloader ========###
    model.eval()
    metrics_dict = model.evaluate(
        dataloader,
        requires_rk=True,
        requires_pk=True,
        requires_cp=False,
        k_test=256,
    )
    print(metrics_dict)




if __name__ == "__main__":
    main()