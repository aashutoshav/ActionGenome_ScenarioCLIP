import torch.nn as nn
from .distillation import DistillationLoss

class TriangularLoss(nn.Module):
    def __init__(self):
        super(TriangularLoss, self).__init__()
        self.distillation_loss_fn = DistillationLoss()

    def forward(self, global_visual_embeddings, global_text_embeddings, object_visual_embeddings, object_text_embeddings, relation_visual_embeddings, relation_text_embeddings):

        global_and_object_visual = self.distillation_loss_fn.divergence(global_visual_embeddings, object_visual_embeddings)
        global_and_relation_visual = self.distillation_loss_fn.divergence(global_visual_embeddings, relation_visual_embeddings)
        relation_and_object_visual = self.distillation_loss_fn.divergence(relation_visual_embeddings, object_visual_embeddings)

        global_and_object_text = self.distillation_loss_fn.divergence(global_text_embeddings, object_text_embeddings)
        global_and_relation_text = self.distillation_loss_fn.divergence(global_text_embeddings, relation_text_embeddings)
        relation_and_object_text = self.distillation_loss_fn.divergence(relation_text_embeddings, object_text_embeddings)

        total_divergence = global_and_object_visual + global_and_relation_visual + relation_and_object_visual + global_and_object_text + global_and_relation_text + relation_and_object_text

        return total_divergence
