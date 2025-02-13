import argparse
import torch
from transformers import AutoImageProcessor

from dataset import (
    MultiFolderDualDataset,
    ImageFolderDualDataset,
    CIFAR100DualDataset,
    dual_collate_fn,
    load_labels
)
from combined_nvila.models.model import DualEncoderWithPE
from train import train_dual_encoder_probe
from test import test_evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="train or test")
    parser.add_argument("--dataset_name", type=str, default="imagenet100",
                        choices=["imagenet100", "cifar100"],
                        help="Which dataset to use: imagenet100 or cifar100")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    # ImageNet100 전용 인자
    parser.add_argument("--train_dirs", nargs='+', default=[])
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--labels_path", type=str, default="")

    # CLIP / DINO
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino_model_name", type=str, default="facebook/dinov2-base")

    # S2 scaling
    parser.add_argument("--default_scale", type=int, default=[224])
    parser.add_argument("--scales", type=int, nargs='+', default=[1])
    # parser.add_argument("--clip_num_prefix", type=int, default=1)
    # parser.add_argument("--dino_num_prefix", type=int, default=1)
    
    # Positional Encoding
    parser.add_argument("--attention_heads", type=int, default=6)

    parser.add_argument("--embed_dim", type=int, default=768)
                        
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--project_name", type=str, default="multis2_transformer")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Processor
    clip_processor = None
    dino_processor = None
    if args.clip_model_name != "":
        clip_processor = AutoImageProcessor.from_pretrained(args.clip_model_name)
    if args.dino_model_name != "":
        dino_processor = AutoImageProcessor.from_pretrained(args.dino_model_name)

    # num_classes 결정
    if args.dataset_name == "imagenet100":
        # ImageNet-100
        id_to_class = load_labels(args.labels_path)
        num_classes = len(id_to_class)
    else:
        # CIFAR-100
        # 100 classes
        num_classes = 100
        id_to_class = None  # CIFAR-100 클래스는 내부 dataset에서만 사용

    # 모델 생성
    model = DualEncoderWithPE(
        clip_model_name=args.clip_model_name,

        dino_model_name=args.dino_model_name,
        
        default_scale = args.default_scale,
        scales = args.scales,

        num_classes=num_classes,
        attention_heads=args.attention_heads,
        embed_dim = args.embed_dim,
    )

    if args.mode == "train":
        # -----------------------------
        # (1) ImageNet-100
        # -----------------------------
        if args.dataset_name == "imagenet100":
            train_dataset = MultiFolderDualDataset(
                folders=args.train_dirs,
                clip_processor=clip_processor,
                dino_processor=dino_processor,
                id_to_class=id_to_class
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=dual_collate_fn
            )

            val_dataset = ImageFolderDualDataset(
                folder=args.test_dir,
                clip_processor=clip_processor,
                dino_processor=dino_processor,
                id_to_class=id_to_class
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=dual_collate_fn
            )

        # -----------------------------
        # (2) CIFAR-100
        # -----------------------------
        else:  # cifar100
            train_dataset = CIFAR100DualDataset(
                split="train",
                clip_processor=clip_processor,
                dino_processor=dino_processor,
                download=True
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=dual_collate_fn
            )

            # validation -> 간단히 test split 사용
            val_dataset = CIFAR100DualDataset(
                split="test",
                clip_processor=clip_processor,
                dino_processor=dino_processor,
                download=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=dual_collate_fn
            )

        # 학습
        best_ckpt = train_dual_encoder_probe(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            project_name=args.project_name,
            checkpoint_dir=args.checkpoint_dir,
            resume_checkpoint=args.resume_checkpoint
        )
        print("Best checkpoint saved at:", best_ckpt)

    elif args.mode == "test":
        # -----------------------------
        # (1) ImageNet-100
        # -----------------------------
        if args.dataset_name == "imagenet100":
            test_dataset = ImageFolderDualDataset(
                folder=args.test_dir,
                clip_processor=clip_processor,
                dino_processor=dino_processor,
                id_to_class=id_to_class
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=dual_collate_fn
            )

        # -----------------------------
        # (2) CIFAR-100
        # -----------------------------
        else:  # cifar100
            test_dataset = CIFAR100DualDataset(
                split="test",
                clip_processor=clip_processor,
                dino_processor=dino_processor,
                download=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=dual_collate_fn
            )

        # checkpoint 로드
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
        else:
            raise ValueError("checkpoint_path must be specified for test mode.")

        test_evaluation(model, test_loader, device)

    else:
        raise ValueError("--mode must be train or test")

if __name__ == "__main__":
    main()
