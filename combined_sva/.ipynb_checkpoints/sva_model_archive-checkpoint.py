import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from transformers import CLIPVisionModel, Dinov2Model


########################################
# 1. SVA 모듈
########################################

class SVA(nn.Module):
    """
    Spatial Vision Aggregator (SVA)
    
    여러 비전 인코더의 특징 맵을, 학습 가능한 2차원 쿼리 토큰(격자)을 이용해 
    크로스 어텐션으로 통합합니다.
    
    Args:
        d_model (int): 특징 차원 (예: 768)
        grid_size (int): 2D 격자 한 변의 크기 (예: 16 -> 16x16=256 쿼리)
        num_encoders (int): 통합할 비전 인코더 수 (예: 2)
        num_heads (int): Multi-head attention의 head 수
    """
    def __init__(self, d_model, grid_size, num_encoders, num_heads=8):
        super(SVA, self).__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.num_queries = grid_size * grid_size  # 총 쿼리 수
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        
        # 학습 가능한 쿼리 토큰 (num_queries x d_model)
        self.queries = nn.Parameter(torch.randn(self.num_queries, d_model))
        
        # 각 인코더의 특징 맵을 각 입력 hidden_dim에 맞춰 d_model로 변환
        # 여기서는 CLIP: 1024 -> d_model, DINO: 768 -> d_model
        self.projections = nn.ModuleList([
            nn.Linear(1024, d_model),  # CLIP 변환
            nn.Linear(768, d_model)    # DINO 변환 (768 -> d_model)
        ])
        
        # Multi-head Attention (batch_first=True)
        self.attn = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
    def forward(self, features_list):
        """
        Args:
            features_list (List[Tensor]): 
                각 원소 shape: (B, C, H, W)
                (예: CLIP의 경우 (B, 1024, H, W), DINO의 경우 (B, 768, H, W))
                
        Returns:
            aggregated_queries: (B, num_queries, d_model)
        """
        B = features_list[0].shape[0]
        aggregated_queries = 0.0
        
        for i, feat in enumerate(features_list):
            # feat: (B, C, H, W)
            # flatten하여 (B, H*W, C)
            B, C, H, W = feat.shape
            feat = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            # Projection: 각 인코더에 맞는 선형 변환 적용 → (B, H*W, d_model)
            feat = self.projections[i](feat)
            
            # 새 채널 차원: d_model
            new_C = feat.shape[2]  # should be d_model
            
            # H, W 자동 계산: (B, H*W, d_model)에서 H*W를 정사각형으로 가정
            total_tokens = feat.shape[1]
            H_new = W_new = int(total_tokens ** 0.5)
            assert H_new * W_new == total_tokens, f"Invalid shape: total_tokens={total_tokens}, expected {H_new}x{W_new}"
            
            # 다시 (B, d_model, H_new, W_new)로 변환
            feat = feat.transpose(1, 2).view(B, new_C, H_new, W_new)
            
            # Unfold를 이용해 grid_size에 맞게 나누고 평균 풀링
            # 여기서 patch 크기는 H_new // grid_size, W_new // grid_size
            patch_h = H_new // self.grid_size
            patch_w = W_new // self.grid_size
            feat = feat.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
            # feat: (B, new_C, grid_size, patch_h, grid_size, patch_w)
            feat = feat.contiguous().view(B, new_C, self.num_queries, -1)
            feat = feat.mean(dim=-1)  # (B, new_C, num_queries)
            feat = feat.transpose(1, 2)  # (B, num_queries, new_C)
            
            # 학습 가능한 쿼리 토큰 확장: (B, num_queries, d_model)
            queries_expanded = self.queries.unsqueeze(0).expand(B, -1, -1)
            attn_output, _ = self.attn(query=queries_expanded, key=feat, value=feat)
            aggregated_queries += attn_output
        
        # 여러 인코더 결과 평균
        aggregated_queries = aggregated_queries / self.num_encoders
        return aggregated_queries


########################################
# 2. 전체 모델
########################################

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
        
        # ---------------------------------
        # CLIP 모델 설정
        # ---------------------------------
        if clip_model_name != "":
            self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_scales = clip_scales
            self.clip_num_prefix = clip_num_prefix
            clip_hidden_dim = self.clip_model.config.hidden_size  # 예: 1024
        else:
            self.clip_model = None
            self.clip_scales = []
            self.clip_num_prefix = 0
            clip_hidden_dim = 0

        # ---------------------------------
        # DINO 모델 설정
        # ---------------------------------
        if dino_model_name != "":
            self.dino_model = Dinov2Model.from_pretrained(dino_model_name)
            for param in self.dino_model.parameters():
                param.requires_grad = False
            self.dino_scales = dino_scales
            self.dino_num_prefix = dino_num_prefix
            dino_hidden_dim = self.dino_model.config.hidden_size  # 예: 768
        else:
            self.dino_model = None
            self.dino_scales = []
            self.dino_num_prefix = 0
            dino_hidden_dim = 0
        
        # ---------------------------------
        # SVA 모듈: CLIP + DINO (총 2개의 인코더)
        # ---------------------------------
        self.sva = SVA(
            d_model=d_model,
            grid_size=grid_size,
            num_encoders=2,
            num_heads=8
        )
        
        # 최종 분류기
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, clip_x, dino_x):
        """
        clip_x, dino_x: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        feature_list = []
        
        # CLIP forward 처리
        if self.clip_model is not None and clip_x is not None:
            clip_outputs = self.clip_model(pixel_values=clip_x)
            # CLS 토큰 제거: (B, seq_len, hidden_dim)
            clip_feat = clip_outputs.last_hidden_state[:, 1:, :]
            B, seq_len, dim = clip_feat.shape
            H = W = int(seq_len ** 0.5)
            assert H * W == seq_len, f"Invalid shape: seq_len={seq_len}, expected {H}x{W}"
            clip_feat = clip_feat.permute(0, 2, 1).reshape(B, dim, H, W)
            feature_list.append(clip_feat)

        # DINO forward 처리
        if self.dino_model is not None and dino_x is not None:
            dino_outputs = self.dino_model(pixel_values=dino_x)
            dino_feat = dino_outputs.last_hidden_state[:, 1:, :]
            B, seq_len, dim = dino_feat.shape
            H = W = int(seq_len ** 0.5)
            assert H * W == seq_len, f"Invalid shape: seq_len={seq_len}, expected {H}x{W}"
            dino_feat = dino_feat.permute(0, 2, 1).reshape(B, dim, H, W)
            feature_list.append(dino_feat)

        if len(feature_list) == 0:
            raise ValueError("No valid features from CLIP or DINO were provided.")

        aggregated = self.sva(feature_list)
        pooled = aggregated.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
