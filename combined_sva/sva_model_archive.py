import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from transformers import CLIPVisionModel, Dinov2Model


########################################
# 1. SVA 모듈 (개선 버전)
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
        
        # ===== 학습 가능한 쿼리 토큰 (num_queries x d_model) =====
        self.queries = nn.Parameter(torch.randn(self.num_queries, d_model))
        
        # ===== 각 인코더별 Projection: Conv2d + GroupNorm =====
        #  - CLIP: 1024 -> d_model
        #  - DINO: 768 -> d_model
        #  - GroupNorm(1, d_model) : 채널 전체를 하나의 그룹으로 묶어 정규화
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
        
        # ===== Multi-head Attention (batch_first=True) =====
        self.attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, features_list):
        """
        Args:
            features_list (List[Tensor]):
                각 원소 shape: (B, C, H, W)
                예) CLIP -> (B, 1024, H, W), DINO -> (B, 768, H, W)
                
        Returns:
            aggregated_queries: (B, num_queries, d_model)
        """
        B = features_list[0].shape[0]
        
        # 크로스 어텐션 결과들을 누적할 텐서
        aggregated_queries = 0.0
        
        # ----------------------------------
        # 여러 인코더의 특징 맵에 대해 반복
        # ----------------------------------
        for i, feat in enumerate(features_list):
            # feat: (B, C, H, W)

            # 1) Conv2d + GroupNorm 적용 -> (B, d_model, H, W)
            feat = self.projections[i](feat)

            # 2) Adaptive Pool을 통해 grid_size x grid_size로 변환
            #    (B, d_model, grid_size, grid_size)
            feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))

            # 3) (B, d_model, grid_size^2) -> (B, grid_size^2, d_model)
            B_, C_, _ = feat.shape[0], feat.shape[1], feat.shape[2] * feat.shape[3]
            feat = feat.view(B_, C_, -1).transpose(1, 2)  # (B, grid_size^2, d_model)

            # 4) 크로스 어텐션을 위한 쿼리
            #    (B, num_queries, d_model)
            queries_expanded = self.queries.unsqueeze(0).expand(B_, -1, -1)

            # 5) Multi-head Attention (Cross Attention)
            #    query: queries_expanded, key/value: feat
            attn_output, _ = self.attn(
                query=queries_expanded,  # (B, grid_size^2, d_model)
                key=feat,                # (B, grid_size^2, d_model)
                value=feat               # (B, grid_size^2, d_model)
            )
            
            # 인코더별 출력 누적
            aggregated_queries += attn_output
        
        # 여러 인코더 결과 평균
        aggregated_queries /= self.num_encoders
        
        return aggregated_queries


########################################
# 2. 전체 모델 (인터페이스 동일)
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
        
        # 최종 분류기 (d_model -> num_classes)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, clip_x, dino_x):
        """
        clip_x, dino_x: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        feature_list = []
        
        # ---------------------------------
        # 1) CLIP Forward
        # ---------------------------------
        if self.clip_model is not None and clip_x is not None:
            clip_outputs = self.clip_model(pixel_values=clip_x)
            # CLS 토큰 제거: (B, seq_len, hidden_dim)
            clip_feat = clip_outputs.last_hidden_state[:, 1:, :]
            B, seq_len, dim = clip_feat.shape
            H = W = int(seq_len ** 0.5)
            assert H * W == seq_len, f"Invalid shape: seq_len={seq_len}, expected {H}x{W}"
            # (B, dim, H, W)
            clip_feat = clip_feat.permute(0, 2, 1).reshape(B, dim, H, W)
            feature_list.append(clip_feat)

        # ---------------------------------
        # 2) DINO Forward
        # ---------------------------------
        if self.dino_model is not None and dino_x is not None:
            dino_outputs = self.dino_model(pixel_values=dino_x)
            dino_feat = dino_outputs.last_hidden_state[:, 1:, :]
            B, seq_len, dim = dino_feat.shape
            H = W = int(seq_len ** 0.5)
            assert H * W == seq_len, f"Invalid shape: seq_len={seq_len}, expected {H}x{W}"
            # (B, dim, H, W)
            dino_feat = dino_feat.permute(0, 2, 1).reshape(B, dim, H, W)
            feature_list.append(dino_feat)

        if len(feature_list) == 0:
            raise ValueError("No valid features from CLIP or DINO were provided.")

        # ---------------------------------
        # 3) SVA: 크로스 어텐션으로 통합
        # ---------------------------------
        aggregated = self.sva(feature_list)   # (B, grid_size^2, d_model)
        
        # ---------------------------------
        # 4) 최종 분류
        # ---------------------------------
        pooled = aggregated.mean(dim=1)       # (B, d_model)
        logits = self.classifier(pooled)      # (B, num_classes)
        
        return logits
