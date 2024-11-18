# MedUnifier

Implementation of MedUnifer in PyTorch and Transformers

## Requisite
see packaeges on environment.yaml

## Usage
1. Pre-training
> accelerate launch --config_file ./config/[FILENAME] train.py [OPTIONS]

2. Fine-tuned classification task
> python -m downstream_tasks.classification.supervised_cls [OPTIONS]

3. Zero-shot classification task
> python -m downstream_tasks.classification.zero_shot [OPTIONS]

4. Retrieval task
> python -m downstream_tasks.retrieval.retrieve [OPTIONS]
