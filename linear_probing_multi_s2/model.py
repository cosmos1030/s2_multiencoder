import torch
import torch.nn as nn
from transformers import CLIPVisionModel, Dinov2Model
from s2wrapper import forward as multiscale_forward

class DualEncoderLinearProbeS2(nn.Module):
    """
    1) CLIP + S2 multiscale
    2) DINO + S2 multiscale
    CLS 임베딩들 concat -> Linear
    """
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        clip_scales=[1],
        clip_num_prefix=1,

        dino_model_name="facebook/dinov2-base",
        dino_scales=[1],
        dino_num_prefix=1,

        num_classes=100
    ):
        super().__init__()
        # CLIP
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        clip_hidden_dim = self.clip_model.config.hidden_size

        self.clip_scales = clip_scales
        self.clip_num_prefix = clip_num_prefix

        # DINO
        self.dino_model = Dinov2Model.from_pretrained(dino_model_name)
        for param in self.dino_model.parameters():
            param.requires_grad = False
        dino_hidden_dim = self.dino_model.config.hidden_size

        self.dino_scales = dino_scales
        self.dino_num_prefix = dino_num_prefix

        # 최종 임베딩 차원
        out_dim = (clip_hidden_dim * len(clip_scales)) + (dino_hidden_dim * len(dino_scales))
        self.linear = nn.Linear(out_dim, num_classes)

    def forward_clip(self, x):
        # CLIP forward -> last_hidden_state
        return self.clip_model(x).last_hidden_state

    def forward_dino(self, x):
        # DINO forward -> last_hidden_state
        return self.dino_model(pixel_values=x).last_hidden_state

    def forward(self, clip_x, dino_x):
        with torch.no_grad():
            # (1) CLIP + S2
            clip_out = multiscale_forward(
                forward_fn=self.forward_clip,
                x=clip_x,
                scales=self.clip_scales,
                num_prefix_token=self.clip_num_prefix
            )  # [B, seq_len, clip_dim * len(scales)]
            clip_cls = clip_out[:, 0, :]

            # (2) DINO + S2
            dino_out = multiscale_forward(
                forward_fn=self.forward_dino,
                x=dino_x,
                scales=self.dino_scales,
                num_prefix_token=self.dino_num_prefix
            )  # [B, seq_len, dino_dim * len(scales)]
            dino_cls = dino_out[:, 0, :]

        # (3) Concat -> Linear
        concat_feat = torch.cat([clip_cls, dino_cls], dim=1)
        logits = self.linear(concat_feat)
        return logits
