import torch
import lightning as L
from torch.optim import AdamW
from transformers import CLIPTokenizer
from losses.contrastive import ContrastiveLoss
from .backbone import OneLevelCLIP

class PyramidCLIP(L.LightningModule):
    def __init__(self, vision_model_name, text_model_name, clip_tokenizer_name, lr=1e-5):
        super().__init__()
        self.model = OneLevelCLIP(vision_model_name, text_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_name)
        self.contrastive_loss_fn = ContrastiveLoss()
        self.lr = lr

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch

        caption_tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        global_visual_embeddings = self.model.encode_image(images)

        global_text_embeddings = self.model.encode_text(caption_tokens.input_ids)

        object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
        object_captions = [caption for cap_list in object_names for caption in cap_list]
        object_tokens = self.tokenizer(object_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        object_visual_embeddings = self.model.encode_image(object_images)
        object_text_embeddings = self.model.encode_text(object_tokens.input_ids)

        relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
        relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
        relation_tokens = self.tokenizer(relation_captions_flattened, return_tensors="pt", padding=True, truncation=True).to(self.device)

        relation_visual_embeddings = self.model.encode_image(relation_images_stacked)
        relation_text_embeddings = self.model.encode_text(relation_tokens.input_ids)

        global_contrastive_loss = self.contrastive_loss_fn(global_visual_embeddings, global_text_embeddings)
        object_contrastive_loss = self.contrastive_loss_fn(object_visual_embeddings, object_text_embeddings)
        relation_contrastive_loss = self.contrastive_loss_fn(relation_visual_embeddings, relation_text_embeddings)

        contrastive_loss = global_contrastive_loss + object_contrastive_loss + relation_contrastive_loss

        self.log_dict({
            'train_contrastive_loss': contrastive_loss
        }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

        return contrastive_loss

    def validation_step(self, batch, batch_idx):
        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch

        caption_tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        global_visual_embeddings = self.model.encode_image(images)

        global_text_embeddings = self.model.encode_text(caption_tokens.input_ids)

        object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
        object_captions = [caption for cap_list in object_names for caption in cap_list]
        object_tokens = self.tokenizer(object_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

        object_visual_embeddings = self.model.encode_image(object_images)
        object_text_embeddings = self.model.encode_text(object_tokens.input_ids)

        relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
        relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
        relation_tokens = self.tokenizer(relation_captions_flattened, return_tensors="pt", padding=True, truncation=True).to(self.device)

        relation_visual_embeddings = self.model.encode_image(relation_images_stacked)
        relation_text_embeddings = self.model.encode_text(relation_tokens.input_ids)

        global_contrastive_loss = self.contrastive_loss_fn(global_visual_embeddings, global_text_embeddings)
        object_contrastive_loss = self.contrastive_loss_fn(object_visual_embeddings, object_text_embeddings)
        relation_contrastive_loss = self.contrastive_loss_fn(relation_visual_embeddings, relation_text_embeddings)

        contrastive_loss = global_contrastive_loss + object_contrastive_loss + relation_contrastive_loss

        self.log_dict({
            'val_contrastive_loss': contrastive_loss
        }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

        return contrastive_loss
