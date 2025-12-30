import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from src.config import (
    BATCH_SIZE, NUM_WORKERS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    CKPT_DIR, CLASSES, VIDEOS_DIR, LABELS_CSV, SEED
)
from src.utils import set_seed
from src.dataset_video_pose_multilabel import read_labels_csv, split_by_videos, VideoPoseMultiLabelDataset
from src.model_twostream import TwoStreamFusionNet

@torch.no_grad()
def evaluate(model, loader, device="cuda", threshold=0.5):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    ys, ps = [], []

    for frames, pose, y in loader:
        frames, pose, y = frames.to(device), pose.to(device), y.to(device)
        logits = model(frames, pose)
        loss = bce(logits, y)

        probs = torch.sigmoid(logits)
        ys.append(y.detach().cpu().numpy())
        ps.append(probs.detach().cpu().numpy())
        total_loss += loss.item()

    y_true = np.vstack(ys)
    y_prob = np.vstack(ps)
    y_pred = (y_prob >= threshold).astype(int)

    return total_loss / max(len(loader), 1), y_true, y_pred

def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    items = read_labels_csv(LABELS_CSV)
    train_items, val_items, test_items = split_by_videos(
        items, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED
    )

    test_ds = VideoPoseMultiLabelDataset(VIDEOS_DIR, test_items, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda")
    )

    ckpt_path = CKPT_DIR / "squats_twostream_gru.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model = TwoStreamFusionNet(num_classes=len(CLASSES), pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    threshold = float(ckpt.get("threshold", 0.5))
    loss, y_true, y_pred = evaluate(model, test_loader, device=device, threshold=threshold)

    print(f"\nTest loss: {loss:.4f} | threshold={threshold}")

    print("\n=== Classification Report ===")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    ))

    print("\n=== Confusion Matrices (per class) ===")
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for i, cls in enumerate(CLASSES):
        print(f"\n{cls}:")
        print(mcm[i])

if __name__ == "__main__":
    main()
