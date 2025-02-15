import torch
import torch.nn as nn
from transformers import CLIPVisionModel, Dinov2Model
from peft import LoraConfig, get_peft_model
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
        use_lora=False,  # Choose whether to use LoRA
        lora_r=8,  
        lora_alpha=16,  
        lora_dropout=0.05  
    ):
        super().__init__()

        self.use_lora = use_lora  # Store LoRA usage flag
        if use_lora:
            print("using lora")

        # ---------------------------------
        # CLIP with Optional LoRA
        # ---------------------------------
        if clip_model_name:
            self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)

            if self.use_lora:
                target_modules = [
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj" for i in range(12)
                ] + [
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj" for i in range(12)
                ] + [
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj" for i in range(12)
                ] + [
                    f"vision_model.encoder.layers.{i}.self_attn.out_proj" for i in range(12)
                ]

                lora_config_clip = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none"
                )
                
                self.clip_model = get_peft_model(self.clip_model, lora_config_clip)


            self.clip_scales = clip_scales
            self.clip_num_prefix = clip_num_prefix
            clip_hidden_dim = self.clip_model.config.hidden_size
        else:
            self.clip_model = None
            self.clip_scales = []
            self.clip_num_prefix = 0
            clip_hidden_dim = 0

        # ---------------------------------
        # DINO with Optional LoRA
        # ---------------------------------
        if dino_model_name:
            self.dino_model = Dinov2Model.from_pretrained(dino_model_name)

            if self.use_lora:
                dino_target_modules = [
                    f"encoder.layer.{i}.attention.attention.query" for i in range(12)
                ] + [
                    f"encoder.layer.{i}.attention.attention.key" for i in range(12)
                ] + [
                    f"encoder.layer.{i}.attention.attention.value" for i in range(12)
                ] + [
                    f"encoder.layer.{i}.attention.output.dense" for i in range(12)
                ]

                lora_config_dino = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=dino_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none"
                )
                self.dino_model = get_peft_model(self.dino_model, lora_config_dino)

            self.dino_scales = dino_scales
            self.dino_num_prefix = dino_num_prefix
            dino_hidden_dim = self.dino_model.config.hidden_size
        else:
            self.dino_model = None
            self.dino_scales = []
            self.dino_num_prefix = 0
            dino_hidden_dim = 0

        # Final embedding dimension
        out_dim = (clip_hidden_dim * len(clip_scales)) + (dino_hidden_dim * len(dino_scales))
        self.linear = nn.Linear(out_dim, num_classes)

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
                )
                clip_cls = clip_out[:, 0, :]
                feats.append(clip_cls)

            # (2) DINO Processing
            if self.dino_model and dino_x is not None:
                dino_out = multiscale_forward(
                    self.forward_dino,
                    dino_x,
                    scales=self.dino_scales,
                    num_prefix_token=self.dino_num_prefix,
                    output_shape="bnc"
                )
                dino_cls = dino_out[:, 0, :]
                feats.append(dino_cls)

        # (3) Concat -> Linear
        concat_feat = torch.cat(feats, dim=1)
        logits = self.linear(concat_feat)
        return logits
