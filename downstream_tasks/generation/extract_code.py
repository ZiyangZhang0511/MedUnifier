import argparse
from tqdm.auto import tqdm
import pickle
import random
from collections import namedtuple

import lmdb

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from pytorch_lightning import seed_everything

from .utils import create_model, save_batch_images
from utilities.utils import build_dataset, build_dataloader
CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--single_sentence", action="store_true")
    parser.add_argument("--save_path", type=str)

    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()

    return args

def extract(lmdb_env, dataloader, model, device):
    index = 0
    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(dataloader)

        filename_id = 0
        for batch in pbar:
            batch["image"] = batch["image"].to(device)

            id_t, id_b = model.get_encoding_ids(batch)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for top, bottom in zip(id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=f"code_{filename_id}")
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')
                filename_id += 1

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))

def main():

    args = get_args()

    ###======== create model ========###
    model = create_model("full_model", args.ckpt_path, args.device)
    model.to(args.device)
    # model.eval()

    ###======== build dataloader ========###
    dataset = build_dataset(dataset_name="new_mimic_cxr_full", mode="train", single_sentence=args.single_sentence)
    dataloader = build_dataloader(dataset, batch_size=args.batch_size, mode="train")
    print(len(dataset))

    ###======== set up lmdb environment ========###
    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(args.save_path, map_size=map_size)
    extract(env, dataloader, model, args.device)

if __name__ == "__main__":
    seed_everything(1)
    main()