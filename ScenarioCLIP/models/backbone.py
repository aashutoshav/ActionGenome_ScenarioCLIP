import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel

class OneLevelCLIP(nn.Module):
    def __init__(self, vision_model_name='openai/clip-vit-base-patch32', text_model_name='openai/clip-vit-base-patch32'):
        super(OneLevelCLIP, self).__init__()

        self.visual_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_model_name)

        self.image_projection = nn.Linear(self.visual_encoder.config.hidden_size, 512)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, 512)

        self.image_norm = nn.LayerNorm(512)
        self.text_norm = nn.LayerNorm(512)

    def encode_image(self, image_tensor):
        image_outputs = self.visual_encoder(pixel_values=image_tensor)
        image_embeds = image_outputs.pooler_output

        image_embeds = self.image_projection(image_embeds)
        image_embeds = self.image_norm(image_embeds)

        return image_embeds

    def encode_text(self, input_ids):
        text_outputs = self.text_encoder(input_ids=input_ids)
        text_embeds = text_outputs.pooler_output

        text_embeds = self.text_projection(text_embeds)
        text_embeds = self.text_norm(text_embeds)

        return text_embeds

class ThreeLevelCLIP(nn.Module):
    def __init__(self, vision_model_name='openai/clip-vit-base-patch32', text_model_name='openai/clip-vit-base-patch32'):
        super(ThreeLevelCLIP, self).__init__()

        self.global_visual_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.global_text_encoder = CLIPTextModel.from_pretrained(text_model_name)

        self.global_image_projection = nn.Linear(self.global_visual_encoder.config.hidden_size, 512)
        self.global_text_projection = nn.Linear(self.global_text_encoder.config.hidden_size, 512)

        self.global_image_norm = nn.LayerNorm(512)
        self.global_text_norm = nn.LayerNorm(512)

        self.object_visual_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.object_text_encoder = CLIPTextModel.from_pretrained(text_model_name)

        self.object_image_projection = nn.Linear(self.object_visual_encoder.config.hidden_size, 512)
        self.object_text_projection = nn.Linear(self.object_text_encoder.config.hidden_size, 512)

        self.object_image_norm = nn.LayerNorm(512)
        self.object_text_norm = nn.LayerNorm(512)

        self.relation_visual_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.relation_text_encoder = CLIPTextModel.from_pretrained(text_model_name)

        self.relation_image_projection = nn.Linear(self.relation_visual_encoder.config.hidden_size, 512)
        self.relation_text_projection = nn.Linear(self.relation_text_encoder.config.hidden_size, 512)

        self.relation_image_norm = nn.LayerNorm(512)
        self.relation_text_norm = nn.LayerNorm(512)

    def global_encode_image(self, image_tensor):
        image_outputs = self.global_visual_encoder(pixel_values=image_tensor)
        image_embeds = image_outputs.pooler_output

        image_embeds = self.global_image_projection(image_embeds)
        image_embeds = self.global_image_norm(image_embeds)

        return image_embeds

    def global_encode_text(self, input_ids):
        text_outputs = self.global_text_encoder(input_ids=input_ids)
        text_embeds = text_outputs.pooler_output

        text_embeds = self.global_text_projection(text_embeds)
        text_embeds = self.global_text_norm(text_embeds)

        return text_embeds

    def object_encode_image(self, image_tensor):
        image_outputs = self.object_visual_encoder(pixel_values=image_tensor)
        image_embeds = image_outputs.pooler_output

        image_embeds = self.object_image_projection(image_embeds)
        image_embeds = self.object_image_norm(image_embeds)

        return image_embeds

    def object_encode_text(self, input_ids):
        text_outputs = self.object_text_encoder(input_ids=input_ids)
        text_embeds = text_outputs.pooler_output

        text_embeds = self.object_text_projection(text_embeds)
        text_embeds = self.object_text_norm(text_embeds)

        return text_embeds

    def relation_encode_image(self, image_tensor):
        image_outputs = self.relation_visual_encoder(pixel_values=image_tensor)
        image_embeds = image_outputs.pooler_output

        image_embeds = self.relation_image_projection(image_embeds)
        image_embeds = self.relation_image_norm(image_embeds)

        return image_embeds

    def relation_encode_text(self, input_ids):
        text_outputs = self.relation_text_encoder(input_ids=input_ids)
        text_embeds = text_outputs.pooler_output

        text_embeds = self.relation_text_projection(text_embeds)
        text_embeds = self.relation_text_norm(text_embeds)

        return text_embeds
