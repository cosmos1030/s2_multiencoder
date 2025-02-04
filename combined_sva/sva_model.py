import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from transformers import CLIPVisionModel, Dinov2Model

class SVA(nn.Module):
    def __init__(self, d_model, grid_size, num_encoders, num_heads=8):
        super(SVA, self).__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.num_queries = grid_size * grid_size
        self.num_encoders = num_encoders
        self.num_heads = num_heads

        # 학습 가능한 2D Query
        self.queries = nn.Parameter(torch.randn(self.num_queries, d_model))

        # Conv2d + GroupNorm(1, d_model)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, d_model, kernel_size=1),
                nn.GroupNorm(1, d_model),
            ),
            nn.Sequential(
                nn.Conv2d(768, d_model, kernel_size=1),
                nn.GroupNorm(1, d_model),
            )
        ])

        self.attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # 여러 인코더 concat -> 최종 변환
        self.final_proj = nn.Linear(d_model * num_encoders, d_model)

    def forward(self, features_list):
        B = features_list[0].shape[0]
        outputs = []

        for i, feat in enumerate(features_list):
            # feat: (B, C, H, W)
            feat = self.projections[i](feat)  # -> (B, d_model, H, W)
            feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
            # (B, d_model, grid_size, grid_size)

            B_, C_, grid_area = feat.shape[0], feat.shape[1], feat.shape[2]*feat.shape[3]
            feat = feat.view(B_, C_, grid_area).permute(0, 2, 1)  # (B, grid_area, d_model)

            queries_expanded = self.queries.unsqueeze(0).expand(B_, -1, -1)
            attn_output, _ = self.attn(queries_expanded, feat, feat)
            outputs.append(attn_output)

        aggregated_queries = torch.cat(outputs, dim=-1)  # (B, grid_area, d_model * num_encoders)
        aggregated_queries = self.final_proj(aggregated_queries)
        return aggregated_queries


class SVA_Model(nn.Module):
    def __init__(
        self, 
        clip_model_name="openai/clip-vit-large-patch14",
        clip_scales=[1.0],
        clip_num_prefix=1,
        dino_model_name="facebook/dinov2-base",
        dino_scales=[1.0],
        dino_num_prefix=1,
        num_classes=100,
        grid_size=16,
        d_model=768
    ):
        super(SVA_Model, self).__init__()
        
        # 인자 보관 (주로 main.py 호환용)
        self.clip_model_name = clip_model_name
        self.clip_scales = clip_scales
        self.clip_num_prefix = clip_num_prefix
        self.dino_model_name = dino_model_name
        self.dino_scales = dino_scales
        self.dino_num_prefix = dino_num_prefix
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.d_model = d_model

        # CLIP
        if self.clip_model_name != "":
            self.clip_model = CLIPVisionModel.from_pretrained(self.clip_model_name)
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            self.clip_model = None

        # DINO
        if self.dino_model_name != "":
            self.dino_model = Dinov2Model.from_pretrained(self.dino_model_name)
            for param in self.dino_model.parameters():
                param.requires_grad = False
        else:
            self.dino_model = None

        # SVA
        self.sva = SVA(
            d_model=self.d_model,
            grid_size=self.grid_size,
            num_encoders=2, 
            num_heads=8
        )

        # 분류기
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, clip_x, dino_x):
        feature_list = []

        # CLIP
        if self.clip_model is not None and clip_x is not None:
            clip_outputs = self.clip_model(pixel_values=clip_x)
            clip_feat = clip_outputs.last_hidden_state[:, 1:, :]  # (B, seq_len-1, 1024)
            B, seq_len, dim = clip_feat.shape
            H = W = int(seq_len**0.5)
            clip_feat = clip_feat.permute(0, 2, 1).reshape(B, dim, H, W)
            feature_list.append(clip_feat)

        # DINO
        if self.dino_model is not None and dino_x is not None:
            dino_outputs = self.dino_model(pixel_values=dino_x)
            dino_feat = dino_outputs.last_hidden_state[:, 1:, :]  # (B, seq_len-1, 768)
            B, seq_len, dim = dino_feat.shape
            H = W = int(seq_len**0.5)
            dino_feat = dino_feat.permute(0, 2, 1).reshape(B, dim, H, W)
            feature_list.append(dino_feat)

        if len(feature_list) == 0:
            raise ValueError("No valid features from CLIP or DINO were provided.")

        aggregated = self.sva(feature_list)  # (B, grid_size^2, d_model)

        pooled = aggregated.mean(dim=1)      # (B, d_model)
        logits = self.classifier(pooled)     # (B, num_classes)
        return logits
