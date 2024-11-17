import torch

@torch.no_grad()
def concat_all_gather(tensor):
    return tensor

def all_gather_with_grad(tensors):
    return tensors