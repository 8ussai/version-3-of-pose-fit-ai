from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

from .config import DATA_DIR, CLASSES, BATCH_SIZE, NUM_WORKERS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, CKPT_DIR
from .utils import set_seed
from .dataset_video_pose import list_videos, split_items, VideoPoseDataset
from .model_twostream import TwoStreamFusionNet


@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    all_y = []
    all_pred = []

    for frames, pose, y in tqdm(loader, leave=False):
        frames = frames.to(device)
        pose = pose.to(device)
        y = y.to(device)

        logits = model(frames, pose)
        loss = ce(logits, y)

        pred = logits.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        correct += (pred == y).sum().item()
        total += y.size(0)

        all_y.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)

    return avg_loss, acc, np.array(all_y), np.array(all_pred)


def main():
    set_seed()

    items = list_videos(DATA_DIR)
    if len(items) == 0:
        raise RuntimeError(f"No videos found under: {DATA_DIR}")

    train_items, val_items, test_items = split_items(items, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    test_ds = VideoPoseDataset(test_items, class_to_idx, training=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = CKPT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}. Train first (python -m src.train).")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = TwoStreamFusionNet(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])

    loss, acc, y_true, y_pred = evaluate(model, test_loader, device=device)

    print(f"\n=== Test Results ===")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}\n")

    # تقرير التصنيف
    print("=== Classification Report ===")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("=== Confusion Matrix (rows=true, cols=pred) ===")
    print(cm)

    # (اختياري) طباعة أسماء الكلاسات لسهولة القراءة
    print("\nClass index mapping:")
    for i in range(len(CLASSES)):
        print(f"{i}: {idx_to_class[i]}")


if __name__ == "__main__":
    main()
