import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import os
import time

def train_dual_encoder_probe(
    model, 
    train_loader, 
    val_loader, 
    epochs, 
    lr, 
    device,
    project_name="my_wandb_project",
    checkpoint_dir="./checkpoints",
    resume_checkpoint=None,
    early_stopping_patience=3
):
    """
    - Val Loss가 개선될 때만 checkpoint 저장
    - 3번 연속 개선 없으면 early stopping
    - 최대 epochs
    - 반환: 최적 모델이 저장된 checkpoint 경로
    """
    wandb.init(project=project_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.linear.parameters(), lr=lr)

    model.to(device)

    best_val_loss = float('inf')
    early_stop_counter = 0
    start_epoch = 1

    # 체크포인트 폴더 생성
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Resume 체크포인트 로드
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        print(f"[INFO] Resume from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1
    else:
        if resume_checkpoint:
            print(f"[WARNING] Checkpoint not found: {resume_checkpoint}, start from scratch.")

    best_ckpt_path = None

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        # ----------------------
        # (1) Training
        # ----------------------
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for clip_imgs, dino_imgs, labels in train_pbar:
            # None인 경우에만 건너뛰고, 아니면 .to(device)
            if clip_imgs is not None:
                clip_imgs = clip_imgs.to(device)
            if dino_imgs is not None:
                dino_imgs = dino_imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(clip_imgs, dino_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            train_pbar.set_postfix(loss=train_loss_sum / (train_total + 1e-6))  # Avoid division by zero


        train_loss_epoch = train_loss_sum / len(train_loader.dataset)
        train_acc_epoch  = train_correct / train_total

        # ----------------------
        # (2) Validation
        # ----------------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for clip_imgs, dino_imgs, labels in val_pbar:
                if clip_imgs is not None:
                    clip_imgs = clip_imgs.to(device)
                if dino_imgs is not None:
                    dino_imgs = dino_imgs.to(device)
                labels = labels.to(device)

                outputs = model(clip_imgs, dino_imgs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * labels.size(0)

                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss_epoch = val_loss_sum / len(val_loader.dataset)
        val_acc_epoch  = val_correct / val_total

        # 결과 출력
        print(f"  Train Loss: {train_loss_epoch:.4f} | Train Acc: {train_acc_epoch:.4f}")
        print(f"  Val   Loss: {val_loss_epoch:.4f} | Val   Acc: {val_acc_epoch:.4f}")

        # W&B 로깅
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "train_acc": train_acc_epoch,
            "val_loss": val_loss_epoch,
            "val_acc": val_acc_epoch
        })

        # ----------------------
        # (3) Early Stopping 체크
        # ----------------------
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            early_stop_counter = 0

            best_ckpt_path = os.path.join(checkpoint_dir, f"best_epoch_{epoch}.pth")
            model_info = {
                "model_type": "Clip+Dino",
                "saved_time": time.strftime("%Y-%m-%d_%H:%M:%S"),
                "epoch": epoch
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "model_info": model_info
            }, best_ckpt_path)

            print(f" [*] Val loss improved. Checkpoint saved to {best_ckpt_path}")
        else:
            early_stop_counter += 1
            print(f" [!] No improvement. Early stop counter: {early_stop_counter}/{early_stopping_patience}")
            if early_stop_counter >= early_stopping_patience:
                print(f" [!!!] Early stopping triggered at epoch {epoch}.")
                break

    wandb.finish()
    return best_ckpt_path
