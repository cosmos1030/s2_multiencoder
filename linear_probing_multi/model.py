import torch.nn as nn
import torch
from transformers import CLIPVisionModel, Dinov2Model

class DualEncoderLinearProbe(nn.Module):
    def __init__(self, clip_model_name, dino_model_name, num_classes=100):
        super().__init__()
        # CLIP
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_hidden_dim = self.clip_model.config.hidden_size  # 예: 768

        # DINOv2
        self.dino_model = Dinov2Model.from_pretrained(dino_model_name)
        dino_hidden_dim = self.dino_model.config.hidden_size  # 예: 768

        # Freeze
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False

        # 최종 Linear
        self.linear = nn.Linear(clip_hidden_dim + dino_hidden_dim, num_classes)

    def forward(self, clip_pixels, dino_pixels):
        with torch.no_grad():
            clip_out = self.clip_model(clip_pixels)
            clip_feat = clip_out.last_hidden_state[:, 0, :]  # CLS 토큰

            dino_out = self.dino_model(pixel_values=dino_pixels)
            dino_feat = dino_out.last_hidden_state[:, 0, :]

        concat_feat = torch.cat([clip_feat, dino_feat], dim=1)
        logits = self.linear(concat_feat)
        return logits
