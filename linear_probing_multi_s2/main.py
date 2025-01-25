import argparse
import torch
from torch.utils.data import DataLoader, random_split

from dataset import (
    MultiFolderDualDataset, 
    ImageFolderDualDataset, 
    load_labels
)
from model import DualEncoderLinearProbe
from train import train_dual_encoder_probe
from test import test_evaluation

def create_datasets_for_train_val(
    train_dirs, 
    labels_path, 
    clip_model_name, 
    dino_model_name,
    val_ratio=0.1
):
    """
    'train.X' 폴더들로부터 전체 Dataset 생성 후,
    일부를 Validation 용으로 분할 (예: 90:10).
    """
    from transformers import AutoProcessor, AutoImageProcessor

    # 라벨 로드
    id_to_class = load_labels(labels_path)
    
    # Processor
    clip_processor = AutoProcessor.from_pretrained(clip_model_name)
    dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)

    # 전체 Train Dataset
    full_dataset = MultiFolderDualDataset(
        folders=train_dirs, 
        clip_processor=clip_processor, 
        dino_processor=dino_processor, 
        id_to_class=id_to_class
    )

    # random_split: 90% train, 10% val
    total_len = len(full_dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(
        dataset=full_dataset, 
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(42)  # 재현성
    )
    return train_dataset, val_dataset, id_to_class


def create_test_dataset(test_dir, labels_path, clip_model_name, dino_model_name):
    """
    실제 Test용 Dataset (ImageFolderDualDataset)
    """
    from transformers import AutoProcessor, AutoImageProcessor

    id_to_class = load_labels(labels_path)
    clip_processor = AutoProcessor.from_pretrained(clip_model_name)
    dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)

    test_dataset = ImageFolderDualDataset(
        folder=test_dir,
        clip_processor=clip_processor,
        dino_processor=dino_processor,
        id_to_class=id_to_class
    )
    return test_dataset, id_to_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------
    # Global Options
    # ------------------
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Choose 'train' to train+val (and optionally test), or 'test' to test only.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino_model_name", type=str, default="facebook/dinov2-base")

    # ------------------
    # Train Options
    # ------------------
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_dirs", type=str, nargs="+", default=[])
    parser.add_argument("--labels_path", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--project_name", type=str, default="my_wandb_project")

    # ------------------
    # Test Options
    # ------------------
    #  - 'test_dir'를 지정하면, 학습 후 자동 테스트 or --mode test 일 때 데이터셋 경로로 사용
    #  - 'checkpoint_path'는 test only 시 로드할 파일
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint for test-only mode")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------
    # (A) Test-only 모드
    #  -> checkpoint 로드 후 Test dataset으로 평가
    # --------------------------------------
    if args.mode == "test":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path must be specified in test mode!")

        # 1) Test dataset 로드
        test_dataset, id_to_class = create_test_dataset(
            test_dir=args.test_dir,
            labels_path=args.labels_path,
            clip_model_name=args.clip_model_name,
            dino_model_name=args.dino_model_name
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # 2) 모델 초기화
        num_classes = len(id_to_class)
        model = DualEncoderLinearProbe(
            clip_model_name=args.clip_model_name,
            dino_model_name=args.dino_model_name,
            num_classes=num_classes
        ).to(device)

        # 3) 체크포인트 로드
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_saved = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", None)
        model_info = checkpoint.get("model_info", {})
        print(f"\n[INFO] Loaded checkpoint from epoch {epoch_saved}, best_val_loss={best_val_loss}, info={model_info}")

        # 4) 테스트 평가
        test_evaluation(model, test_loader, device=device)

        exit(0)  # 종료

    # --------------------------------------
    # (B) Train+Val 모드
    #  -> random_split으로 train/val
    #     EarlyStopping & Checkpoint
    #     훈련 종료 후 Test dataset 평가(옵션)
    # --------------------------------------
    # (1) Train/Val dataset
    print(args.train_dirs)
    train_dataset, val_dataset, id_to_class = create_datasets_for_train_val(
        train_dirs=args.train_dirs,
        labels_path=args.labels_path,
        clip_model_name=args.clip_model_name,
        dino_model_name=args.dino_model_name,
        val_ratio=0.1
    )
    print(len(train_dataset), len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # (2) 모델 생성
    num_classes = len(id_to_class)
    model = DualEncoderLinearProbe(
        clip_model_name=args.clip_model_name,
        dino_model_name=args.dino_model_name,
        num_classes=num_classes
    )

    # (3) 학습 (Early Stopping & Val Loss 기준으로 Best 체크포인트 저장)
    best_ckpt_path = train_dual_encoder_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        project_name=args.project_name,
        checkpoint_dir=args.checkpoint_dir,
        resume_checkpoint=args.resume_checkpoint,
        early_stopping_patience=3
    )

    print("\n[*] Training/Validation Finished.")

    # (4) 학습 끝나면, Test셋으로 최종 성능 측정
    if args.test_dir:
        print("[*] Now evaluating on test set using best checkpoint.")
        # Test dataset 만들기
        test_dataset, _ = create_test_dataset(
            test_dir=args.test_dir,
            labels_path=args.labels_path,
            clip_model_name=args.clip_model_name,
            dino_model_name=args.dino_model_name
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Best 체크포인트 로드
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_saved = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        model_info = checkpoint.get("model_info", {})
        print(f"[INFO] Loaded best checkpoint from epoch {epoch_saved}, best_val_loss={best_val_loss}, info={model_info}")

        # 테스트 평가
        test_evaluation(model, test_loader, device=device)
