import torch
import torch.nn as nn
from transformers import CLIPVisionModel, Dinov2Model
from s2wrapper import forward as multiscale_forward

class DualEncoderLinearProbe(nn.Module):
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

        # ---------------------------------
        # CLIP
        # ---------------------------------
        if clip_model_name != "":
            self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_scales = clip_scales
            self.clip_num_prefix = clip_num_prefix
            clip_hidden_dim = self.clip_model.config.hidden_size
        else:
            self.clip_model = None
            self.clip_scales = []
            self.clip_num_prefix = 0
            clip_hidden_dim = 0

        # ---------------------------------
        # DINO
        # ---------------------------------
        if dino_model_name != "":
            self.dino_model = Dinov2Model.from_pretrained(dino_model_name)
            for param in self.dino_model.parameters():
                param.requires_grad = False
            self.dino_scales = dino_scales
            self.dino_num_prefix = dino_num_prefix
            dino_hidden_dim = self.dino_model.config.hidden_size
        else:
            self.dino_model = None
            self.dino_scales = []
            self.dino_num_prefix = 0
            dino_hidden_dim = 0

        # 최종 Linear 입력 차원
        out_dim = (clip_hidden_dim * len(self.clip_scales)) + (dino_hidden_dim * len(self.dino_scales))

        self.linear = nn.Linear(out_dim, num_classes)

    def forward_clip(self, x):
        # CLIP forward
        return self.clip_model(x).last_hidden_state  # [B, seq_len, dim]

    def forward_dino(self, x):
        # DINO forward
        return self.dino_model(pixel_values=x).last_hidden_state

    def forward(self, clip_x, dino_x):
        """
        clip_x, dino_x 각각 None일 수 있으므로 체크
        """
        feats = []
        with torch.no_grad():
            # (1) CLIP
            if (self.clip_model is not None) and (clip_x is not None):
                clip_out = multiscale_forward(
                    self.forward_clip,
                    clip_x,
                    scales=self.clip_scales,
                    num_prefix_token=self.clip_num_prefix,
                    output_shape="bnc"
                )
                # clip_out: [B, seq_len, clip_dim * len(scales)]
                clip_cls = clip_out[:, 0, :]
                feats.append(clip_cls)

            # (2) DINO
            if (self.dino_model is not None) and (dino_x is not None):
                dino_out = multiscale_forward(
                    self.forward_dino,
                    dino_x,
                    scales=self.dino_scales,
                    num_prefix_token=self.dino_num_prefix,
                    output_shape="bnc"
                )
                dino_cls = dino_out[:, 0, :]
                feats.append(dino_cls)

        if len(feats) == 0:
            raise ValueError("No model input provided (both clip_x and dino_x are None).")

        concat_feat = torch.cat(feats, dim=1)  # [B, out_dim]
        logits = self.linear(concat_feat)      # [B, num_classes]
        return logits
