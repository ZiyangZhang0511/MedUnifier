import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    build_fine_tune_dataset,
    build_fine_tune_dataloader, 
    create_model,
    get_zero_shot_probilities,
    binary_classification_metrics,
    multiclass_classification_metrics,
    binary_multiclass_classification_metrics,
)


class Classifier(nn.Module):

    def __init__(self, pretrained_model_config, num_class, hidden_dim=768):
        super(Classifier, self).__init__()

        self.pretrained_model_config = pretrained_model_config
        self.pretrained_model = create_model(**pretrained_model_config)
        
        if pretrained_model_config["model_type"] in ["full_model", "incomplete_model"]:
            num_query_tokens = self.pretrained_model.query_tokens.size(1)
            qfomer_hidden_dim = self.pretrained_model.query_tokens.size(2)
            img_feat_dim = num_query_tokens * qfomer_hidden_dim

        self.top_classifier = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.Linear(hidden_dim, num_class),
        )


    def forward(self, images):

        bs = images.size(0)
        if self.pretrained_model_config["model_type"] in ["full_model", "incomplete_model"]:
            query_output, _ = self.pretrained_model.forward_image(images)
            img_features = query_output.view(bs, -1)


        logits = self.top_classifier(img_features)

        return logits
    
    @torch.no_grad()
    def evaluate(self, dataloader, dataset_name, device="cuda"):
        if dataset_name in ["rsna_full", "siim"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        epoch_loss = 0
        logits_all = []
        labels_all = []
        for batch in dataloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            logits = self.forward(images)

            loss = criterion(logits, labels)

            epoch_loss += loss.item() / len(dataloader)

            logits_all.append(logits)
            labels_all.append(labels)

        logits_all = torch.cat(logits_all).detach().cpu()
        labels_all = torch.cat(labels_all).detach().cpu()

    
        if dataset_name in ["rsna_full", "siim"]:
            probs_all = torch.sigmoid(logits_all)
            metrics_dict = binary_classification_metrics(probs_all.numpy(), labels_all.numpy())
        else:
            probs_all = F.softmax(logits_all, dim=-1)
            metrics_dict = multiclass_classification_metrics(probs_all.numpy(), labels_all.numpy())
        
        metrics_dict["loss"] = epoch_loss

        return metrics_dict
