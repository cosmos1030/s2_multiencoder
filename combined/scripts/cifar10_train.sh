#!/bin/bash

    # --------------------------
    # CIFAR-10 학습
    # --------------------------
    python main.py \
      --mode train \
      --dataset_name cifar100 \
      --batch_size 8 \
      --epochs 10 \
      --lr 1e-3 \
      --clip_model_name "openai/clip-vit-base-patch32" \
      --dino_model_name "facebook/dinov2-base" \
      --clip_scales 1.0 2.0 \
      --dino_scales 1.0 2.0 \
      --clip_num_prefix 1 \
      --dino_num_prefix 1 \
      --checkpoint_dir "./checkpoints/cifar100/both_s2" \
      --resume_checkpoint "" \
      --project_name "basic_cifar100"


# tqdm으로 Progress Bar 추가
with tqdm(range(max_iter), desc="Training", unit="step") as pbar:
    for step in pbar:
        optimizer.zero_grad()
        loss_value = loss_fn(data_aug, init_params)
        loss_value.backward()
        optimizer.step()

        current_loss = loss_value.item()

        # tqdm의 진행바에 Loss 값 실시간 표시
        pbar.set_postfix(loss=f"{current_loss:.6f}")

        # Tolerance 기준 수렴 체크
        if abs(prev_loss - current_loss) < tolerance:
            print(f"Converged at step {step} with loss={current_loss:.6f}")
            break
        prev_loss = current_loss

print("Final Loss:", prev_loss)