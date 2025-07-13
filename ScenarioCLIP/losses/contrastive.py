import torch
import torch.nn as nn
import lightning as L

class ContrastiveLoss(L.LightningModule):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T) / self.temperature
        labels = torch.arange(len(image_features)).to(logits.device)
        return nn.CrossEntropyLoss()(logits, labels)
