import torch
import torch.nn as nn
from transformers import CLIPVisionModel, Dinov2Model
from s2wrapper import forward as multiscale_forward

class Model(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        clip_scales=[1],
        clip_num_prefix=1,

        dino_model_name="facebook/dinov2-base",
        dino_scales=[1],
        dino_num_prefix=1,

        num_classes=100,
        embed_dim=1536,  # Combined dimension of CLIP + DINO
        num_heads=6,
        num_layers=2
    ):
        super().__init__()

        # ---------------------------------
        # CLIP
        # ---------------------------------
        if clip_model_name:
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
        if dino_model_name:
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

        # ---------------------------------
        # Transformer Encoder
        # ---------------------------------
        self.input_dim = embed_dim*2
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final Classification Layer
        self.linear = nn.Linear(self.input_dim, num_classes)

    def forward_clip(self, x):
        return self.clip_model(x).last_hidden_state  # [B, seq_len, dim]

    def forward_dino(self, x):
        return self.dino_model(pixel_values=x).last_hidden_state  # [B, seq_len, dim]

    def forward(self, clip_x, dino_x):
        feats = []
        with torch.no_grad():
            # (1) CLIP Processing
            if self.clip_model and clip_x is not None:
                clip_out = multiscale_forward(
                    self.forward_clip,
                    clip_x,
                    scales=self.clip_scales,
                    num_prefix_token=self.clip_num_prefix,
                    output_shape="bnc"
                )  # [B, seq_len, clip_dim * len(scales)]
                feats.append(clip_out)
                #print("shape after S2 clip: ", clip_out.shape)

            # (2) DINO Processing
            if self.dino_model and dino_x is not None:
                dino_out = multiscale_forward(
                    self.forward_dino,
                    dino_x,
                    scales=self.dino_scales,
                    num_prefix_token=self.dino_num_prefix,
                    output_shape="bnc"
                )  # [B, seq_len, dino_dim * len(scales)]
                feats.append(dino_out)
                #print("shape after S2 dino: ", dino_out.shape)

        if not feats:
            raise ValueError("No model input provided (both clip_x and dino_x are None).")

        # (3) Concatenate Token Sequences
        concat_feat = torch.cat(feats, dim=1)  # [B, seq_len, input_dim]
        #print("shape after concat: ", concat_feat.shape)

        # (4) Transformer Encoding
        transformed_feat = self.transformer_encoder(concat_feat)  # [B, seq_len, input_dim]

        # (5) Use Mean Pooling Over All Tokens
        pooled_feat = transformed_feat.mean(dim=1)  # [B, input_dim]

        # (6) Final Classification
        logits = self.linear(pooled_feat)  # [B, num_classes]
        return logits
