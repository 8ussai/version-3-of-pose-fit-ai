import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (PROJECT_ROOT, OUTPUT_DIR, CKPT_DIR, CLASSES, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
                     NUM_WORKERS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
from .utils import set_seed, ensure_dir
from .model_twostream import TwoStreamFusionNet
from .dataset_video_pose_multilabel import (
    read_labels_csv, split_by_videos, VideoPoseMultiLabelDataset
)

LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "squats_labels.csv"
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos" / "squats_all"

def batch_f1_micro(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).float()
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()
    f1 = (2*tp) / (2*tp + fp + fn + 1e-9)
    return f1.item()

def run_epoch(model, loader, optimizer=None, device="cuda"):
    train = optimizer is not None
    model.train(train)

    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_f1 = 0.0
    n_batches = 0

    for frames, pose, y in tqdm(loader, leave=False):
        frames = frames.to(device)
        pose = pose.to(device)
        y = y.to(device)  # (B,C) float

        logits = model(frames, pose)        # (B,C)
        loss = loss_fn(logits, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        probs = torch.sigmoid(logits)
        f1 = batch_f1_micro(y, probs, thr=0.5)

        total_loss += loss.item()
        total_f1 += f1
        n_batches += 1

    return total_loss / max(1, n_batches), total_f1 / max(1, n_batches)

def main():
    set_seed()
    ensure_dir(CKPT_DIR)

    items = read_labels_csv(LABELS_CSV)
    if len(items) == 0:
        raise RuntimeError(f"No rows in labels CSV: {LABELS_CSV}")

    train_items, val_items, test_items = split_by_videos(items, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    train_ds = VideoPoseMultiLabelDataset(VIDEOS_DIR, train_items, training=True)
    val_ds   = VideoPoseMultiLabelDataset(VIDEOS_DIR, val_items, training=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoStreamFusionNet(num_classes=len(CLASSES)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = -1.0
    best_path = CKPT_DIR / "best_multilabel.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, optimizer, device)
        va_loss, va_f1 = run_epoch(model, val_loader, None, device)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train F1={tr_f1:.3f} loss={tr_loss:.4f} | val F1={va_f1:.3f} loss={va_loss:.4f}")

        if va_f1 > best_val:
            best_val = va_f1
            torch.save({
                "model_state": model.state_dict(),
                "classes": CLASSES,
                "threshold": 0.5
            }, best_path)
            print(f"✅ Saved best -> {best_path} (val F1={best_val:.3f})")

    print("Done.")

if __name__ == "__main__":
    main()
