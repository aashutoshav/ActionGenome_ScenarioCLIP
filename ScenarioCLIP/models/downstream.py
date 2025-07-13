import os
import json
import torch
import typing
import functools
import torch.nn as nn
import lightning as L
from torch.optim import AdamW

class CLIPFinetune(L.LightningModule):
    def __init__(self, model, tokenizer, classes_json, lr=1e-5, classify: typing.Literal["action", "object", "relation"] = "action"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.lr = lr
        self.classes_json = classes_json
        self.classify = classify
        self.num_classes = 0
        if self.classify == "action":
            self.num_classes = len(self.action_classlist())
        elif self.classify == "object":
            self.num_classes = len(self.object_classlist())
        elif self.classify == "relation":
            self.num_classes = len(self.relation_classlist())
        self.classification_layer = nn.Linear(in_features=512, out_features=self.num_classes)

    @functools.cache
    def action_classlist(self):
        with open(self.classes_json) as f:
            classes = json.load(f)
        return classes['actions']

    @functools.cache
    def object_classlist(self):
        with open(self.classes_json) as f:
            classes = json.load(f)
        return classes['objects']

    @functools.cache
    def relation_classlist(self):
        with open(self.classes_json) as f:
            classes = json.load(f)
        return classes['relations']

    def configure_optimizers(self):
        return AdamW(list(self.model.parameters()) + list(self.classification_layer.parameters()), lr=self.lr)

    def training_step(self, batch, batch_idx):

        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch

        sizes = [len(captions), [len(object_list) for object_list in objects_cropped], [len(relation_list) for relation_list in relation_captions]]

        # Action Recognition

        if self.classify == "action":
            caption_tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

            global_visual_embeddings = self.model.global_encode_image(images)
            frozen_global_visual_embeddings = self.model.frozen_global_encode_image(images)

            text_embeddings = self.model.global_encode_text(caption_tokens.input_ids)

        # Object Level

        elif self.classify == "object":

            object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
            object_captions = [caption for cap_list in object_names for caption in cap_list]
            object_tokens = self.tokenizer(object_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

            object_visual_embeddings = self.model.object_encode_image(object_images)
            object_text_embeddings = self.model.object_encode_text(object_tokens.input_ids)

            frozen_object_text_embeddings = self.model.frozen_object_encode_text(object_tokens.input_ids)

        # Relation Level

        relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
        relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
        relation_tokens = self.tokenizer(relation_captions_flattened, return_tensors="pt", padding=True, truncation=True).to(self.device)

        relation_visual_embeddings = self.model.relation_encode_image(relation_images_stacked)
        relation_text_embeddings = self.model.relation_encode_text(relation_tokens.input_ids)

        frozen_relation_text_embeddings = self.model.frozen_relation_encode_text(relation_tokens.input_ids)

        # Loss Calculation

        if self.contrastive_only: # No KD
            global_contrastive_loss = self.contrastive_loss_fn(global_visual_embeddings, text_embeddings)
            object_contrastive_loss = self.contrastive_loss_fn(object_visual_embeddings, object_text_embeddings)
            relation_contrastive_loss = self.contrastive_loss_fn(relation_visual_embeddings, relation_text_embeddings)

            contrastive_loss = global_contrastive_loss + object_contrastive_loss + relation_contrastive_loss

            self.log_dict({
                'train_contrastive_loss': contrastive_loss
            }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

            return contrastive_loss

        else:
            contrastive_loss, distill_loss = self.total_loss_fn(
                global_visual_embeddings, frozen_global_visual_embeddings, text_embeddings,
                object_visual_embeddings, object_text_embeddings, frozen_object_text_embeddings,
                relation_visual_embeddings, relation_text_embeddings, frozen_relation_text_embeddings,
                sizes
            )

            total_loss = contrastive_loss + distill_loss

            self.log_dict({
                'train_contrastive_loss': contrastive_loss,
                'train_distill_loss': distill_loss,
                'train_total_loss': total_loss,
            }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

            return total_loss

    def validation_step(self, batch, batch_idx):

        # Global Level

        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch

        sizes = [len(captions), [len(object_list) for object_list in objects_cropped], [len(relation_list) for relation_list in relation_captions]]

        caption_tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        global_visual_embeddings = self.model.global_encode_image(images)
        frozen_global_visual_embeddings = self.model.frozen_global_encode_image(images)

        text_embeddings = self.model.global_encode_text(caption_tokens.input_ids)

        # Object Level

        object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
        object_captions = [caption for cap_list in object_names for caption in cap_list]
        object_tokens = self.tokenizer(object_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        object_visual_embeddings = self.model.object_encode_image(object_images)
        object_text_embeddings = self.model.object_encode_text(object_tokens.input_ids)

        frozen_object_text_embeddings = self.model.frozen_object_encode_text(object_tokens.input_ids)

        # Relation Level

        relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
        relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
        relation_tokens = self.tokenizer(relation_captions_flattened, return_tensors="pt", padding=True, truncation=True).to(self.device)

        relation_visual_embeddings = self.model.relation_encode_image(relation_images_stacked)
        relation_text_embeddings = self.model.relation_encode_text(relation_tokens.input_ids)

        frozen_relation_text_embeddings = self.model.frozen_relation_encode_text(relation_tokens.input_ids)

        # Loss Calculation

        if self.contrastive_only: # No KD
            global_contrastive_loss = self.contrastive_loss_fn(global_visual_embeddings, text_embeddings)
            object_contrastive_loss = self.contrastive_loss_fn(object_visual_embeddings, object_text_embeddings)
            relation_contrastive_loss = self.contrastive_loss_fn(relation_visual_embeddings, relation_text_embeddings)

            contrastive_loss = global_contrastive_loss + object_contrastive_loss + relation_contrastive_loss

            self.log_dict({
                'val_contrastive_loss': contrastive_loss
            }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

            return contrastive_loss

        else:
            contrastive_loss, distill_loss = self.total_loss_fn(
                global_visual_embeddings, frozen_global_visual_embeddings, text_embeddings,
                object_visual_embeddings, object_text_embeddings, frozen_object_text_embeddings,
                relation_visual_embeddings, relation_text_embeddings, frozen_relation_text_embeddings,
                sizes
            )

            total_loss = contrastive_loss + distill_loss

            self.log_dict({
                'val_contrastive_loss': contrastive_loss,
                'val_distill_loss': distill_loss,
                'val_total_loss': total_loss,
            }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

            return total_loss

    def on_test_start(self):

        print(f"Storing action embeddings...")

        os.makedirs(f"{self.embedding_storage_dir}/actions", exist_ok=True)

        # Store action embeddings

        for i, action in enumerate(self.action_classlist()):
            caption_tokens = self.tokenizer(action, return_tensors="pt", padding=True, truncation=True).to(self.device)
            action_embedding = self.model.global_encode_text(torch.unsqueeze(caption_tokens.input_ids, 0))
            torch.save(action_embedding, f"{self.embedding_storage_dir}/actions/action_{i}.pt")

        self.correct_predictions = 0
        self.incorrect_predictions = 0

    def test_step(self, batch, batch_idx):

        # Action Recognition

        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch

        image_embeddings = self.model.global_encode_image(images)

        for i, image_embedding in enumerate(image_embeddings):
            similarities = []
            image_embedding = torch.unsqueeze(image_embedding, dim=0)
            for j, action in enumerate(self.action_classlist()):
                action_embedding = torch.load(f"{self.embedding_storage_dir}/actions/action_{j}.pt")
                similarity = self.cosine_similarity(image_embedding, action_embedding)
                similarities.append(similarity.item())
            similarities= torch.Tensor(similarities)
            action_id = torch.argmax(similarities)
            if self.action_classlist()[action_id] == captions[i]:
                self.correct_predictions += 1
            else:
                self.incorrect_predictions += 1

        object_embeddings = self.model.object_encode_image(objects_cropped)

        for i, object_embedding in enumerate(object_embeddings):
            similarities = []
            object_embedding = torch.unsqueeze(image_embedding, dim=0)
            for j, action in enumerate(self.action_classlist()):
                action_embedding = torch.load(f"{self.embedding_storage_dir}/actions/action_{j}.pt")
                similarity = self.cosine_similarity(image_embedding, action_embedding)
                similarities.append(similarity.item())
            similarities= torch.Tensor(similarities)
            action_id = torch.argmax(similarities)
            if self.action_classlist()[action_id] == captions[i]:
                self.correct_predictions += 1
            else:
                self.incorrect_predictions += 1

        image_embeddings = self.model.relation_encode_image(relation_images)

        for i, image_embedding in enumerate(image_embeddings):
            similarities = []
            image_embedding = torch.unsqueeze(image_embedding, dim=0)
            for j, action in enumerate(self.action_classlist()):
                action_embedding = torch.load(f"{self.embedding_storage_dir}/actions/action_{j}.pt")
                similarity = self.cosine_similarity(image_embedding, action_embedding)
                similarities.append(similarity.item())
            similarities= torch.Tensor(similarities)
            action_id = torch.argmax(similarities)
            if self.action_classlist()[action_id] == captions[i]:
                self.correct_predictions += 1
            else:
                self.incorrect_predictions += 1

    def on_test_end(self):
        print(f"Test Predictions: {self.correct_predictions} Correct, {self.incorrect_predictions}: Incorrect")
        print(f"Accuracy = {self.correct_predictions / (self.correct_predictions + self.incorrect_predictions)}")
