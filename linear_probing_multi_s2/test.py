import torch

def test_evaluation(model, test_loader, device):
    """
    모델 + 테스트 로더로 최종 성능(Accuracy 등) 측정
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for clip_imgs, dino_imgs, labels in test_loader:
            clip_imgs = clip_imgs.to(device)
            dino_imgs = dino_imgs.to(device)
            labels    = labels.to(device)

            outputs = model(clip_imgs, dino_imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total if total > 0 else 0
    print(f"[*] Test Accuracy: {test_acc:.4f}")
    return test_acc
