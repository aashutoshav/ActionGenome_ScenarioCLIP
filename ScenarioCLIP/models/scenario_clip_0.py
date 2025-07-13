import os
import json
import torch
import functools
import torch.nn as nn
import lightning as L
import torchmetrics as T
from torch.optim import AdamW
from transformers import CLIPTokenizer
from losses.contrastive import ContrastiveLoss
from .backbone import ThreeLevelCLIP

class ScenarioCLIP0(L.LightningModule):
    def __init__(self, vision_model_name, text_model_name, clip_tokenizer_name, lr=1e-5, embedding_storage_dir=None, classes_json=None, action_test=True, object_test=False, relation_test=False):
        super().__init__()
        self.model = ThreeLevelCLIP(vision_model_name, text_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_name)
        self.contrastive_loss_fn = ContrastiveLoss()
        self.lr = lr
        if embedding_storage_dir:
            self.embedding_storage_dir = embedding_storage_dir
            self.classes_json = classes_json
        self.cosine_similarity = T.CosineSimilarity(reduction='mean')
        self.action_test = action_test
        self.object_test = object_test
        self.relation_test = relation_test

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):

        # Global Level

        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch

        caption_tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        global_visual_embeddings = self.model.global_encode_image(images)

        global_text_embeddings = self.model.global_encode_text(caption_tokens.input_ids)

        # Object Level

        object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
        object_captions = [caption for cap_list in object_names for caption in cap_list]
        object_tokens = self.tokenizer(object_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        object_visual_embeddings = self.model.object_encode_image(object_images)
        object_text_embeddings = self.model.object_encode_text(object_tokens.input_ids)

        # Relation Level

        relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
        relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
        relation_tokens = self.tokenizer(relation_captions_flattened, return_tensors="pt", padding=True, truncation=True).to(self.device)

        relation_visual_embeddings = self.model.relation_encode_image(relation_images_stacked)
        relation_text_embeddings = self.model.relation_encode_text(relation_tokens.input_ids)

        # Loss Calculation

        global_contrastive_loss = self.contrastive_loss_fn(global_visual_embeddings, global_text_embeddings)
        object_contrastive_loss = self.contrastive_loss_fn(object_visual_embeddings, object_text_embeddings)
        relation_contrastive_loss = self.contrastive_loss_fn(relation_visual_embeddings, relation_text_embeddings)

        contrastive_loss = global_contrastive_loss + object_contrastive_loss + relation_contrastive_loss

        self.log_dict({
            'train_contrastive_loss': contrastive_loss
        }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

        return contrastive_loss

    def validation_step(self, batch, batch_idx):

        # Global Level

        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch

        caption_tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        global_visual_embeddings = self.model.global_encode_image(images)

        global_text_embeddings = self.model.global_encode_text(caption_tokens.input_ids)

        # Object Level

        object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
        object_captions = [caption for cap_list in object_names for caption in cap_list]
        object_tokens = self.tokenizer(object_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        object_visual_embeddings = self.model.object_encode_image(object_images)
        object_text_embeddings = self.model.object_encode_text(object_tokens.input_ids)

        # Relation Level

        relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
        relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
        relation_tokens = self.tokenizer(relation_captions_flattened, return_tensors="pt", padding=True, truncation=True).to(self.device)

        relation_visual_embeddings = self.model.relation_encode_image(relation_images_stacked)
        relation_text_embeddings = self.model.relation_encode_text(relation_tokens.input_ids)

        # Loss Calculation

        global_contrastive_loss = self.contrastive_loss_fn(global_visual_embeddings, global_text_embeddings)
        object_contrastive_loss = self.contrastive_loss_fn(object_visual_embeddings, object_text_embeddings)
        relation_contrastive_loss = self.contrastive_loss_fn(relation_visual_embeddings, relation_text_embeddings)

        contrastive_loss = global_contrastive_loss + object_contrastive_loss + relation_contrastive_loss

        self.log_dict({
            'val_contrastive_loss': contrastive_loss
        }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

        return contrastive_loss

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

    def on_test_start(self):

        # Store action embeddings

        if self.action_test:
            print(f"Storing action embeddings...")

            os.makedirs(f"{self.embedding_storage_dir}/actions", exist_ok=True)

            for i, action in enumerate(self.action_classlist()):
                caption_tokens = self.tokenizer(action, return_tensors="pt", padding=True, truncation=True).to(self.device)
                action_embedding = self.model.global_encode_text(torch.unsqueeze(caption_tokens.input_ids, 0))
                torch.save(action_embedding, f"{self.embedding_storage_dir}/actions/action_{i}.pt")

            self.correct_action_predictions = 0
            self.incorrect_action_predictions = 0

        # Store object embeddings

        if self.object_test:
            print(f"Storing object embeddings...")

            os.makedirs(f"{self.embedding_storage_dir}/objects", exist_ok=True)

            for i, object in enumerate(self.object_classlist()):
                caption_tokens = self.tokenizer(object, return_tensors="pt", padding=True, truncation=True).to(self.device)
                object_embedding = self.model.global_encode_text(torch.unsqueeze(caption_tokens.input_ids, 0))
                torch.save(object_embedding, f"{self.embedding_storage_dir}/objects/object_{i}.pt")

            self.correct_object_predictions = 0
            self.incorrect_object_predictions = 0

        # Store relation embeddings

        if self.relation_test:
            print(f"Storing relation embeddings...")

            os.makedirs(f"{self.embedding_storage_dir}/relations", exist_ok=True)

            for i, relation in enumerate(self.relation_classlist()):
                caption_tokens = self.tokenizer(relation, return_tensors="pt", padding=True, truncation=True).to(self.device)
                relation_embedding = self.model.global_encode_text(torch.unsqueeze(caption_tokens.input_ids, 0))
                torch.save(relation_embedding, f"{self.embedding_storage_dir}/relations/relation_{i}.pt")

            self.correct_relation_predictions = 0
            self.incorrect_relation_predictions = 0

    def test_step(self, batch, batch_idx):

        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch

        # Action Recognition

        if self.action_test:
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
                    self.correct_action_predictions += 1
                else:
                    self.incorrect_action_predictions += 1

        # Object Recognition

        if self.object_test:
            object_embeddings = self.model.object_encode_image(objects_cropped)

            for i, object_embedding in enumerate(object_embeddings):
                similarities = []
                object_embedding = torch.unsqueeze(object_embedding, dim=0)
                for j, object in enumerate(self.action_classlist()):
                    object_caption_embedding = torch.load(f"{self.embedding_storage_dir}/objects/object_{j}.pt")
                    similarity = self.cosine_similarity(object_embedding, object_caption_embedding)
                    similarities.append(similarity.item())
                similarities = torch.Tensor(similarities)
                object_id = torch.argmax(similarities)
                if self.object_classlist()[object_id] in object_names[i]:
                    self.correct_object_predictions += 1
                else:
                    self.incorrect_object_predictions += 1

        # Relation Recognition

        if self.relation_test:
            relation_embeddings = self.model.relation_encode_image(relation_images)

            for i, relation_embedding in enumerate(relation_embeddings):
                similarities = []
                relation_embedding = torch.unsqueeze(relation_embedding, dim=0)
                for j, relation in enumerate(self.relation_classlist()):
                    relation_caption_embedding = torch.load(f"{self.embedding_storage_dir}/relations/relation_{j}.pt")
                    similarity = self.cosine_similarity(relation_embedding, relation_caption_embedding)
                    similarities.append(similarity.item())
                similarities= torch.Tensor(similarities)
                relation_id = torch.argmax(similarities)
                if self.relation_classlist()[relation_id] in relation_captions[i]:
                    self.correct_relation_predictions += 1
                else:
                    self.incorrect_relation_predictions += 1

    def on_test_end(self):
        if self.action_test:
            print(f"Action Predictions: {self.correct_action_predictions} Correct, {self.incorrect_action_predictions} Incorrect")
            print(f"Action Recognition Accuracy = {self.correct_action_predictions / (self.correct_action_predictions + self.incorrect_action_predictions)}")
        if self.object_test:
            print(f"Object Predictions: {self.correct_object_predictions} Correct, {self.incorrect_object_predictions} Incorrect")
            print(f"Object Recognition Accuracy = {self.correct_object_predictions / (self.correct_object_predictions + self.incorrect_object_predictions)}")
        if self.relation_test:
            print(f"Relation Predictions: {self.correct_relation_predictions} Correct, {self.incorrect_relation_predictions} Incorrect")
            print(f"Relation Recognition Accuracy = {self.correct_relation_predictions / (self.correct_relation_predictions + self.incorrect_relation_predictions)}")
