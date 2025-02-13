import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, Dinov2Model
from s2wrapper import forward as multiscale_forward

class DualEncoderMultiScale(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        clip_scales=[224, 448],  # CLIP 다중 해상도
        clip_num_prefix=1,

        dino_model_name="facebook/dinov2-base",
        dino_scales=[224, 448],  # DINO 다중 해상도
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

        # 최종 임베딩 차원 (여러 해상도를 고려)
        out_dim = (clip_hidden_dim * len(clip_scales)) + (dino_hidden_dim * len(dino_scales))
        self.linear = nn.Linear(out_dim, num_classes)

    def forward_clip(self, x):
        return self.clip_model(x).last_hidden_state  # CLIP feature

    def forward_dino(self, x):
        return self.dino_model(pixel_values=x).last_hidden_state  # DINO feature
    
    def reshape_feature(self, feat):
        # `seq_len`을 `H, W`로 변환
        H = W = int(feat.shape[1] ** 0.5)  # (보통 정사각형 패치라 가정)
        feat = feat[:, 1:, :]  # CLS 토큰 제거 (필요 시)

        # (B, seq_len, C) → (B, C, H, W)
        feat = feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[2], H, W)
        return feat

    def merge_features_for_multiscale(self, feature_list, output_size):
        """ 여러 해상도의 feature map을 병합하는 함수 """
        resized_features = [
            F.interpolate(feat.to(torch.float32), size=output_size, mode="area").to(feat.dtype)
            for feat in feature_list
        ]
        return torch.cat(resized_features, dim=1)  # (B, C*num_scales, H, W)

    def forward(self, clip_x, dino_x):
        with torch.no_grad():
            # (1) CLIP 다중 해상도 처리
            clip_features = multiscale_forward(
                self.forward_clip, clip_x, scales=self.clip_scales, num_prefix_token=self.clip_num_prefix
            )  # [B, seq_len, clip_dim * len(scales)]

        # ✅ 수정: `torch.split()`은 리스트 반환 -> 개별 변환 필요
            clip_features = [self.reshape_feature(f) for f in torch.split(clip_features, self.clip_model.config.hidden_size, dim=-1)]
        
            # ✅ 수정: `clip_features`가 리스트이므로 `merge_features_for_multiscale` 호출 가능
            clip_merged = self.merge_features_for_multiscale(clip_features, output_size=(1, 1))  
            clip_cls = clip_merged.squeeze(-1).squeeze(-1)  # (B, clip_dim*scales)

            # (2) DINO 다중 해상도 처리
            dino_features = multiscale_forward(
                self.forward_dino, dino_x, scales=self.dino_scales, num_prefix_token=self.dino_num_prefix
            )

            # ✅ 수정: `dino_features`도 동일하게 처리
            dino_features = [self.reshape_feature(f) for f in torch.split(dino_features, self.dino_model.config.hidden_size, dim=-1)]
            dino_merged = self.merge_features_for_multiscale(dino_features, output_size=(1, 1))
            dino_cls = dino_merged.squeeze(-1).squeeze(-1)

        # (3) 병합 -> Linear 예측
        concat_feat = torch.cat([clip_cls, dino_cls], dim=1)  # (B, feature_dim)
        logits = self.linear(concat_feat)
        return logits
