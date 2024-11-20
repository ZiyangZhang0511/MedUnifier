
class Constants:

    ### pre-training ###
    new_mimic_cxr_5x200_dir = "./data/pretrain_data/mimic_cxr_5x200/"
    new_mimic_cxr_full_dir = "./data/pretrain_data/mimic_cxr_full/"

    ### zero-shot classification ###
    chexpert_5x200_dir = "./data/zero_shot_cls_data/chexpert_5x200/"
    rsna_1000_dir = "./data/zero_shot_cls_data/rsna_1000/"

    ### fine-tuning classification ###
    rsna_full_dir = "./data/finetune_cls_data/rsna/"
    chexpert_full_dir = "./data/finetune_cls_data/chexpert_full/"
    covidx_dir = "./data/finetune_cls_data/covidx/"
    siim_dir = "./data/finetune_cls_data/siim/"

CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    'Pneumonia': {
        'severity': ['round', 'early', 'focal', 'multifocal', 'small', ''],
        'subtype': ['bacterial', 'viral', 'mycoplasma '],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left middle lobe",
            "at the right middle lobe",
            ""
        ]
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
        "location": ["left", "right", "tiny"],
    },
}


RSNA_CLASS_PROMPTS = {
    'Pneumonia': {
        "location": [
            "mid lung zone",
            "upper lung zone",
            "right lung zone",
            "left lung zone",
            "lung bases",
            "right lung base",
            "left lung base",
            "bilateral lung bases",
            "left upper lobe",
            "right upper lobe",
            "left lower lobe",
            "right lower lobe",
            "left middle lobe",
            "right middle lobe",
        ],
        'severity': ['multifocal', 'basilar', ''],
        'subtype': ['pneumonia'],

    },
    'Normal': {
        'severity': ['no infection', 'normal', 'negative findings'],
        'subtype': [''],
        "location": [
            "at the left lobe",
            "at the right lobe",
            "at the upper lobe",
            "at the lower lobe",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            ''
        ]
    },
}
