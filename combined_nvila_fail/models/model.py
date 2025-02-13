import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, Dinov2Model
from PIL import Image
import torchvision.transforms as transforms

# S2 Scaling 구현 (CIFAR-100 해상도에 맞게 조정)
def find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size):
    return min(target_ratios, key=lambda x: abs((x[0] / x[1]) - aspect_ratio))

# Tensor → PIL 변환기
to_pil = transforms.ToPILImage()

def dynamic_s2_preprocess(image, default_scale=224, scales=[1,2], max_num=12, image_size=64):
    s2_scales = list(map(lambda x: x*default_scale, scales))

    # PyTorch Tensor이면 PIL 이미지로 변환
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:  # Batch 포함 (ex: [16, 3, 224, 224])
            image = image[0]  # 첫 번째 이미지만 변환
        image = to_pil(image)  # Tensor → PIL 변환

    # 이제 PIL 이미지를 다룰 수 있음
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    min_num = (s2_scales[-1] // s2_scales[0]) ** 2

    processed_images = []

    # 1️⃣ 고정된 정사각형 비율을 사용하여 S2 타일 생성
    for scale in s2_scales[:-1]:
        target_width = image_size * (scale // s2_scales[0])
        target_height = image_size * (scale // s2_scales[0])
        blocks = (scale // s2_scales[0]) ** 2

        resized_img = image.resize((target_width, target_height))
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

    return processed_images, (orig_width, orig_height)


# 좌표 기반 Positional Encoding 정의 (CIFAR-100에 맞게 크기 조정)
class PositionalEncoding(nn.Module):
    def __init__(self, num_positions=64, embed_dim=768):  # 64로 변경
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, num_positions, embed_dim))
        self.num_positions = num_positions
        self.embed_dim = embed_dim

    def forward(self, embeddings, height, width):
        position_embeddings = self.position_embedding[:, :embeddings.shape[1], :]
        scale_factor = (height / self.num_positions, width / self.num_positions)
        patch_pos_embed = position_embeddings.reshape(1, int(self.num_positions**0.5), int(self.num_positions**0.5), self.embed_dim)
        interpolated_pe = F.interpolate(patch_pos_embed, scale_factor=scale_factor, mode="bilinear")
        interpolated_pe = interpolated_pe.reshape(1, embeddings.shape[1], self.embed_dim)
        return embeddings + interpolated_pe


# CLIP + DINO Dual Encoder 모델 정의
class DualEncoderWithPE(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        dino_model_name="facebook/dinov2-base",
        default_scale = 224,
        scales = [1,2],
        num_classes=100,
        attention_heads=8,
        embed_dim = 768
    ):
        super().__init__()
        self.default_scale = default_scale
        self.scales = scales
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        clip_hidden_dim = self.clip_model.config.hidden_size

        self.dino_model = Dinov2Model.from_pretrained(dino_model_name)
        for param in self.dino_model.parameters():
            param.requires_grad = False
        dino_hidden_dim = self.dino_model.config.hidden_size

        # Feature Projection Layer 추가 (차원이 다르면 변환)
        self.clip_proj = nn.Linear(clip_hidden_dim, embed_dim) if clip_hidden_dim != embed_dim else nn.Identity()
        self.dino_proj = nn.Linear(dino_hidden_dim, embed_dim) if dino_hidden_dim != embed_dim else nn.Identity()

        self.pe = PositionalEncoding(num_positions=default_scale, embed_dim=clip_hidden_dim)  # CIFAR-100에 맞춰 수정

        # 가중치 학습을 위한 learnable weights
        self.clip_weight = nn.Parameter(torch.ones(1))
        self.dino_weight = nn.Parameter(torch.ones(1))

        # Self-Attention 적용
        self.attention = nn.MultiheadAttention(embed_dim=clip_hidden_dim + dino_hidden_dim, num_heads=attention_heads)

        # 최종 임베딩 차원 (Self-Attention 후 Linear)
        self.linear = nn.Linear(clip_hidden_dim + dino_hidden_dim, num_classes)

        # 이미지 변환 (CIFAR-100에 맞게 64x64)
        self.transform = transforms.Compose([
            transforms.Resize((default_scale, default_scale)),  # 64x64로 수정
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process_images(self, images):
        """ S2 Scaling 후 텐서 변환 """
        return torch.stack([self.transform(img) for img in images])

    def forward_clip(self, x, height, width):
        clip_features = self.clip_model(x).last_hidden_state
        clip_features = self.pe(clip_features, height, width)
        clip_features = self.clip_proj(clip_features)  # Projection 적용
        return clip_features

    def forward_dino(self, x, height, width):
        dino_features = self.dino_model(pixel_values=x).last_hidden_state
        dino_features = self.pe(dino_features, height, width)
        dino_features = self.dino_proj(dino_features)  # Projection 적용
        return dino_features

    def forward(self, clip_imgs, dino_imgs):
        clip_processed_images, _ = dynamic_s2_preprocess(clip_imgs, default_scale=self.default_scale, scales=self.scales)
        clip_image_tensors = self.process_images(clip_processed_images).to(next(self.parameters()).device)

        dino_processed_images, _ = dynamic_s2_preprocess(dino_imgs, default_scale=self.default_scale, scales=self.scales)
        dino_image_tensors = self.process_images(dino_processed_images).to(next(self.parameters()).device)

        with torch.no_grad():
            clip_feat = self.forward_clip(clip_image_tensors, self.default_scale, self.default_scale)
            dino_feat = self.forward_dino(dino_image_tensors, self.default_scale, self.default_scale)

        # Weighted Feature Fusion
        weighted_clip = self.clip_weight * clip_feat.mean(dim=0)
        weighted_dino = self.dino_weight * dino_feat.mean(dim=0)
        fusion_feat = torch.cat([weighted_clip, weighted_dino], dim=0)

        # Self-Attention을 통한 Feature Fusion
        fusion_feat = fusion_feat.unsqueeze(0)
        attended_feat, _ = self.attention(fusion_feat, fusion_feat, fusion_feat)
        attended_feat = attended_feat.squeeze(0)

        # Linear Classifier 적용
        logits = self.linear(attended_feat.unsqueeze(0))
        return logits
