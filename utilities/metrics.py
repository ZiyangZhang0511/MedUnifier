import torch
import torch.nn.functional as F

import numpy as np
import scipy

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def compute_recall_at_k(similarity_matrix, k):
    N = similarity_matrix.shape[0]
    correct_matches = 0
    
    for i in range(N):
        sorted_indices = np.argsort(-similarity_matrix[i])
        
        if i in sorted_indices[:k]:
            correct_matches += 1
            
    return correct_matches / N
    

def compute_precK(score_matrix, mm_dict, k=1):
    numcases = score_matrix.shape[0]
    res = []
    
    for index, score in enumerate(score_matrix):
        inds = np.argsort(score)[::-1]
        correct_indices = mm_dict[index]

        p = 0.0
        r = 0.0
        for j in range(k):
            if inds[j] in correct_indices:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)


def compute_image_captioning_metric(pred_captions, gt_captions):

    N = len(pred_captions)
    predictions = {i: [pred_captions[i]] for i in range(N)}

    if isinstance(gt_captions[0], str):
        ground_truths = {i: [gt_captions[i]] for i in range(N)}
    elif isinstance(gt_captions[0], list):
        ground_truths = {i: [*gt_captions[i]] for i in range(N)}

    B4 = Bleu(n=4)
    C = Cider()
    M = Meteor()
    R = Rouge()

    # Calculate BLEU score
    b4_score, _ = B4.compute_score(ground_truths, predictions)
    c_score, _ = C.compute_score(ground_truths, predictions)
    m_score, _ = M.compute_score(ground_truths, predictions)
    r_score, _ = R.compute_score(ground_truths, predictions)

    return {
        "B@1": b4_score[0],
        "B@2": b4_score[1],
        "B@3": b4_score[2],
        "B@4": b4_score[3],
        "CIDEr": c_score,
        "METEOR": m_score,
        "ROUGE_L": r_score
    }
