import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class DistillationLoss(L.LightningModule):
    def __init__(self, temperature=0.07):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')  # KL divergence loss

    def one_teacher_many_students(self, teacher_logits, student_group):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = 0.

        for i in range(len(student_group)):
            student_probs = F.log_softmax(student_group[i] / self.temperature, dim=-1)
            distill_loss += self.kl_div_loss(student_probs, teacher_probs) * (self.temperature ** 2)

        distill_loss = distill_loss / len(student_group)

        return distill_loss

    def one_student_many_teachers(self, student_logits, teacher_group):
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        distill_loss = 0.

        for i in range(len(teacher_group)):
            teacher_probs = F.log_softmax(teacher_group[i] / self.temperature, dim=-1)
            distill_loss += self.kl_div_loss(teacher_probs, student_probs) * (self.temperature ** 2)

        distill_loss = distill_loss / len(teacher_group)

        return distill_loss

    def divergence(self, embedding1, embedding2):
        if len(embedding1.shape) == 1:
            embedding1 = torch.unsqueeze(embedding1, dim=0)
        if len(embedding2.shape) == 1:
            embedding2 = torch.unsqueeze(embedding2, dim=0)
        embedding1 = F.log_softmax(embedding1 / self.temperature, dim=0)
        embedding2 = F.log_softmax(embedding2 / self.temperature, dim=0)
        kl = 0.
        for i in range(embedding1.shape[0]):
            for j in range(embedding2.shape[0]):
                kl_1_2 = self.kl_div_loss(embedding1[i], embedding2[j])
                kl_2_1 = self.kl_div_loss(embedding2[j], embedding1[i])
                kl += kl_1_2 + kl_2_1
        return kl
